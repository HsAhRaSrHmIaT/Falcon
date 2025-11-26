import logging
import json
# import os
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Dict, Any

from dotenv import load_dotenv
from livekit.agents import ( #type: ignore
    Agent,
    AgentSession,
    JobContext,
    JobProcess,
    MetricsCollectedEvent,
    RoomInputOptions,
    WorkerOptions,
    cli,
    metrics,
    tokenize,
    function_tool,
    RunContext
)
from livekit.plugins import murf, silero, google, deepgram, noise_cancellation #type: ignore
from livekit.plugins.turn_detector.multilingual import MultilingualModel #type: ignore
from livekit.agents import ChatContext #type: ignore

logger = logging.getLogger("agent")

load_dotenv(".env")

# Configuration constants - can be moved to env vars or config file
CONFIG = {
    "stt_model": "nova-3",
    "llm_model": "gemini-2.5-flash",
    "voice": "en-US-alicia",  # Professional female voice for SDR
    "company_data_file": Path(__file__).parent.parent / "shared-data" / "razorpay_company_data.json",
    "leads_directory": Path(__file__).parent.parent / "leads"
}


@dataclass
class LeadData:
    """Data structure for collecting lead information."""
    name: Optional[str] = None
    company: Optional[str] = None
    email: Optional[str] = None
    role: Optional[str] = None
    use_case: Optional[str] = None
    team_size: Optional[str] = None
    timeline: Optional[str] = None
    notes: Optional[str] = None
    call_start_time: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert lead data to dictionary for JSON serialization."""
        return {
            'name': self.name,
            'company': self.company,
            'email': self.email,
            'role': self.role,
            'use_case': self.use_case,
            'team_size': self.team_size,
            'timeline': self.timeline,
            'notes': self.notes,
            'call_start_time': self.call_start_time,
            'call_end_time': datetime.now().isoformat()
        }
    
    def get_completion_status(self) -> Dict[str, bool]:
        """Get which fields have been collected."""
        return {
            'name': self.name is not None,
            'company': self.company is not None,
            'email': self.email is not None,
            'role': self.role is not None,
            'use_case': self.use_case is not None,
            'team_size': self.team_size is not None,
            'timeline': self.timeline is not None
        }
    
    def get_missing_fields(self) -> list[str]:
        """Get list of missing required fields."""
        status = self.get_completion_status()
        return [field for field, completed in status.items() if not completed]


@dataclass
class SDRData:
    """Shared data for the SDR session."""
    company_info: Dict[str, Any] = field(default_factory=dict)
    faq_data: list[Dict[str, str]] = field(default_factory=list)
    lead_data: LeadData = field(default_factory=LeadData)
    conversation_stage: str = "greeting"  # greeting, discovery, qualification, closing
    call_summary: Optional[str] = None


class BaseAgent(Agent):
    """Base agent class with common initialization."""
    
    def __init__(self, instructions: str, chat_ctx: Optional[ChatContext] = None) -> None:
        super().__init__(
            instructions=instructions,
            chat_ctx=chat_ctx,
            stt=deepgram.STT(model=CONFIG["stt_model"]),
            llm=google.LLM(model=CONFIG["llm_model"]),
            tts=murf.TTS(
                voice=CONFIG["voice"],
                style="Conversation",
                tokenizer=tokenize.basic.SentenceTokenizer(min_sentence_len=2),
                text_pacing=True
            ),
            vad=silero.VAD.load(),
        )


class SDRAgent(BaseAgent):
    """Sales Development Representative agent for lead qualification and company information."""
    
    def __init__(self, chat_ctx: Optional[ChatContext] = None) -> None:
        instructions = """You are Aurora, a friendly and professional Sales Development Representative at Razorpay. Pronounce

        Your role is to:
        1. Warmly greet visitors and introduce yourself and Razorpay
        2. Discover what brought them to Razorpay and what business challenges they're facing
        3. Answer questions about Razorpay's products, services, and pricing using the FAQ knowledge
        4. Naturally collect lead qualification information during the conversation
        5. Provide value and build rapport while understanding their business needs
        6. Create a summary when they're ready to end the call

        CRITICAL: When the user provides any of the following information, you MUST immediately call the collect_lead_info function:
        - Name: call collect_lead_info(field="name", value=their_name)
        - Company/Business name: call collect_lead_info(field="company", value=company_name)
        - Email: call collect_lead_info(field="email", value=email_address)
        - Role/Position: call collect_lead_info(field="role", value=their_role)
        - Use case/what they want: call collect_lead_info(field="use_case", value=their_needs)
        - Team size/employees: call collect_lead_info(field="team_size", value=team_info)
        - Timeline: call collect_lead_info(field="timeline", value=timeline_info)

        Lead Information to Collect (do this naturally during conversation):
        - Name and company
        - Email address
        - Role/position
        - Use case (what they want to use Razorpay for - payments, banking, loans, payroll)
        - Team/business size
        - Timeline for implementation

        Always be helpful, professional, and focused on understanding their business needs. 
        Use the FAQ data to answer questions accurately and avoid making up information not in the knowledge base.
        
        When someone says they're done or ready to end the call, provide a summary and save their information."""

        super().__init__(instructions=instructions, chat_ctx=chat_ctx)

    @function_tool
    async def search_faq(self, context: RunContext[SDRData], question: str) -> str:
        """Search FAQ for relevant answers to user questions."""
        question_lower = question.lower()
        best_match = None
        best_score = 0
        
        # Simple keyword matching for FAQ search
        for faq_item in context.userdata.faq_data:
            faq_question = faq_item['question'].lower()
            faq_answer = faq_item['answer'].lower()
            
            # Count common words between user question and FAQ
            question_words = set(question_lower.split())
            faq_words = set(faq_question.split() + faq_answer.split())
            common_words = question_words.intersection(faq_words)
            
            if len(common_words) > best_score:
                best_score = len(common_words)
                best_match = faq_item
        
        if best_match and best_score > 0:
            return f"Based on our information: {best_match['answer']}"
        else:
            # Fallback to general company information
            company_info = context.userdata.company_info
            return f"I'd be happy to help with information about {company_info['company']['name']}. {company_info['company']['description']} Could you be more specific about what you'd like to know?"

    @function_tool
    async def collect_lead_info(self, context: RunContext[SDRData], field: str, value: str) -> str:
        """Collect and store lead information."""
        lead = context.userdata.lead_data
        
        if field == "name":
            lead.name = value
            return f"Great to meet you, {value}! "
        elif field == "company":
            lead.company = value
            return f"Excellent! Tell me more about {value}. "
        elif field == "email":
            lead.email = value
            return "Perfect! I'll make sure to follow up with you. "
        elif field == "role":
            lead.role = value
            return f"Interesting! As a {value}, I can see how Razorpay's fintech solutions might fit into your work. "
        elif field == "use_case":
            lead.use_case = value
            return "That's exactly the kind of business challenge Razorpay can help solve! "
        elif field == "team_size":
            lead.team_size = value
            return "That context helps me understand your scale. "
        elif field == "timeline":
            lead.timeline = value
            return "Good to know your timeline. "
        else:
            # Store any other information in notes
            if lead.notes:
                lead.notes += f"; {field}: {value}"
            else:
                lead.notes = f"{field}: {value}"
            return "I've noted that information. "

    @function_tool
    async def check_lead_completeness(self, context: RunContext[SDRData]) -> str:
        """Check what lead information is still needed."""
        lead = context.userdata.lead_data
        missing_fields = lead.get_missing_fields()
        
        if not missing_fields:
            return "Great! I have all your key information. Is there anything else you'd like to know about Razorpay?"
        
        # Naturally ask for missing information
        if "name" in missing_fields:
            return "I'd love to personalize our conversation - what's your name?"
        elif "company" in missing_fields:
            return "What company are you with?"
        elif "email" in missing_fields:
            return "What's the best email to reach you at for follow-up?"
        elif "role" in missing_fields:
            return "What's your role at your company?"
        elif "use_case" in missing_fields:
            return "What brings you to Razorpay? What business challenges are you looking to solve?"
        elif "team_size" in missing_fields:
            return "Can you tell me about the size of your business or team?"
        elif "timeline" in missing_fields:
            return "What's your timeline for implementing a payment or fintech solution?"
        
        return "Is there anything else you'd like to know about how Razorpay can help your business?"

    @function_tool
    async def create_call_summary(self, context: RunContext[SDRData]) -> str:
        """Generate and save a call summary with lead information."""
        lead = context.userdata.lead_data
        company_info = context.userdata.company_info
        
        # Create summary
        summary_parts = []
        if lead.name:
            summary_parts.append(f"Spoke with {lead.name}")
        if lead.company and lead.role:
            summary_parts.append(f"who is a {lead.role} at {lead.company}")
        elif lead.company:
            summary_parts.append(f"from {lead.company}")
        elif lead.role:
            summary_parts.append(f"who works as a {lead.role}")
            
        if lead.use_case:
            summary_parts.append(f"They're interested in: {lead.use_case}")
        
        if lead.timeline:
            summary_parts.append(f"Timeline: {lead.timeline}")
            
        if lead.team_size:
            summary_parts.append(f"Team size: {lead.team_size}")
        
        summary = ". ".join(summary_parts) + "."
        context.userdata.call_summary = summary
        
        # Save lead data to file
        try:
            leads_dir = CONFIG["leads_directory"]
            leads_dir.mkdir(exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"lead_razorpay_{timestamp}.json"
            filepath = leads_dir / filename
            
            lead_data_dict = lead.to_dict()
            lead_data_dict["call_summary"] = summary
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(lead_data_dict, f, indent=2)
                
            logger.info(f"Lead data saved to: {filepath}")
            
        except Exception as e:
            logger.error(f"Error saving lead data: {e}")
        
        # Return verbal summary
        if summary_parts:
            return f"Perfect! Let me summarize: {summary} Thanks for your time today! I'll follow up with you soon with more information about how Razorpay can help your business grow."
        else:
            return "Thanks for your interest in Razorpay! I'll follow up with you soon with more information."

    @function_tool
    async def get_company_overview(self, context: RunContext[SDRData]) -> str:
        """Provide an overview of Razorpay's services."""
        company_info = context.userdata.company_info
        company = company_info["company"]
        products = company_info["products"]
        
        overview = f"{company['name']} - {company['tagline']}. "
        overview += f"{company['description']} "
        overview += f"We offer several key solutions: "
        
        product_descriptions = []
        for product_key, product_info in products.items():
            product_descriptions.append(f"{product_info['name']} for {product_info['description']}")
        
        overview += ", ".join(product_descriptions)
        overview += ". How can we help your business with payments and financial solutions?"
        
        return overview


def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()
    proc.userdata["vad"] = silero.VAD.load()


async def entrypoint(ctx: JobContext):
    # Logging setup
    ctx.log_context_fields = {
        "room": ctx.room.name,
    }

    # Initialize SDR data
    userdata = SDRData()
    
    # Load company data
    company_data_file = CONFIG["company_data_file"]
    try:
        if company_data_file.exists():
            with open(company_data_file, 'r', encoding='utf-8') as f:
                company_data = json.load(f)
                userdata.company_info = company_data
                userdata.faq_data = company_data.get("faq", [])
                logger.info(f"Loaded company data for {company_data['company']['name']}")
        else:
            logger.error(f"Company data file not found: {company_data_file}")
            userdata.company_info = {}
            userdata.faq_data = []
    except Exception as e:
        logger.error(f"Error loading company data: {e}")
        userdata.company_info = {}
        userdata.faq_data = []
    
    # Create SDR agent
    sdr_agent = SDRAgent()
    
    # Create the session with the SDR agent
    session = AgentSession[SDRData](
        userdata=userdata,
        stt=deepgram.STT(model="nova-3"),
        llm=google.LLM(model="gemini-2.5-flash"),
        vad=silero.VAD.load(),
        turn_detection=MultilingualModel(),
        preemptive_generation=True,
    )

    # Metrics collection
    usage_collector = metrics.UsageCollector()

    @session.on("metrics_collected")
    def _on_metrics_collected(ev: MetricsCollectedEvent):
        metrics.log_metrics(ev.metrics)
        usage_collector.collect(ev.metrics)

    async def log_usage():
        summary = usage_collector.get_summary()
        logger.info(f"Usage: {summary}")

    ctx.add_shutdown_callback(log_usage)

    # Start the session with the SDR agent
    await session.start(
        agent=sdr_agent,
        room=ctx.room,
        room_input_options=RoomInputOptions(
            noise_cancellation=noise_cancellation.BVC(),
        ),
    )

    # Join the room and connect to the user
    await ctx.connect()


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))
