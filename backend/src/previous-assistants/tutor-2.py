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
    "voices": {
        "tutor": "en-US-matthew",
        "learn": "en-US-matthew",
        "quiz": "en-US-alicia",
        "teach_back": "en-US-ken"
    },
    "content_file": Path(__file__).parent.parent / "shared-data" / "day4_tutor_content.json",
    "key_terms": {
        'variables': ['store', 'value', 'container', 'label', 'reuse'],
        'loops': ['repeat', 'iterate', 'for', 'while', 'condition'],
        'functions': ['reusable', 'parameter', 'input', 'return', 'block'],
        'conditionals': ['if', 'condition', 'decision', 'true', 'false', 'else']
    }
}


@dataclass
class TutorData:
    """Shared data for the tutoring session."""
    concepts: Dict[str, Dict[str, Any]] = field(default_factory=dict)  # id -> concept dict
    current_mode: Optional[str] = None  # 'learn', 'quiz', 'teach_back'
    current_concept: Optional[Dict[str, Any]] = None


class BaseAgent(Agent):
    """Base agent class with common initialization and transfer logic."""
    
    def __init__(self, instructions: str, voice_key: str, chat_ctx: Optional[ChatContext] = None) -> None:
        super().__init__(
            instructions=instructions,
            chat_ctx=chat_ctx,
            stt=deepgram.STT(model=CONFIG["stt_model"]),
            llm=google.LLM(model=CONFIG["llm_model"]),
            tts=murf.TTS(
                voice=CONFIG["voices"][voice_key],
                style="Conversation",
                tokenizer=tokenize.basic.SentenceTokenizer(min_sentence_len=2),
                text_pacing=True
            ),
            vad=silero.VAD.load(),
        )

    async def _transfer_to_agent(self, mode: str, context: RunContext[TutorData]) -> Agent:
        """Transfer to the specified mode agent."""
        userdata = context.userdata
        next_agent = userdata.personas[mode]
        userdata.prev_agent = self
        return next_agent


class TutorAgent(BaseAgent):
    """Main agent that greets users and routes to learning modes."""
    
    def __init__(self, chat_ctx: Optional[ChatContext] = None) -> None:
        instructions = """You are a friendly Active Recall Coach that helps users learn programming concepts.

        Your role is to:
        1. Greet the user warmly and ask which learning mode they'd like to use
        2. Explain the three modes: learn (explanations), quiz (questions), teach_back (teaching back)
        3. Transfer them to the appropriate specialized agent based on their choice
        4. Allow users to switch modes at any time by asking

        Available modes:
        - learn: Get clear explanations of programming concepts
        - quiz: Test your understanding with questions  
        - teach_back: Explain concepts back to receive feedback

        Be welcoming and encouraging. Users can switch modes anytime by saying things like "switch to quiz mode" or "I want to learn about variables"."""

        super().__init__(instructions=instructions, voice_key="tutor", chat_ctx=chat_ctx)

    @function_tool
    async def transfer_to_learn_mode(self, context: RunContext[TutorData]) -> Agent:
        """Transfer to learn mode agent."""
        await self.session.say("Great! I'll connect you with our Learn agent who will explain concepts clearly.")
        return await self._transfer_to_agent("learn", context)

    @function_tool
    async def transfer_to_quiz_mode(self, context: RunContext[TutorData]) -> Agent:
        """Transfer to quiz mode agent."""
        await self.session.say("Excellent! I'll connect you with our Quiz agent who will test your understanding.")
        return await self._transfer_to_agent("quiz", context)

    @function_tool
    async def transfer_to_teach_back_mode(self, context: RunContext[TutorData]) -> Agent:
        """Transfer to teach back mode agent."""
        await self.session.say("Perfect! I'll connect you with our Teach Back agent who will help you practice explaining concepts.")
        return await self._transfer_to_agent("teach_back", context)


class LearnAgent(BaseAgent):
    """Agent for explaining programming concepts."""
    
    def __init__(self, chat_ctx: Optional[ChatContext] = None) -> None:
        instructions = """You are Matthew, a warm and instructional tutor focused on explaining programming concepts clearly.

        CRITICAL: When you first take over from another agent, IMMEDIATELY start with this exact introduction:
        "Hi! I'm Matthew, your learning guide! I'm here to explain programming concepts in simple, easy-to-understand terms. What concept would you like to explore today?"

        Your role is to:
        - Always start with the introduction above when you first take control
        - Explain programming concepts in simple, beginner-friendly terms
        - Use the provided concept summaries from the content file
        - Ask follow-up questions to deepen understanding
        - Encourage the user and be supportive
        - Allow users to switch to other modes when they're ready

        Available concepts: variables, loops, functions, conditionals

        Be patient and thorough in your explanations. When users want to switch modes, transfer them appropriately."""

        super().__init__(instructions=instructions, voice_key="learn", chat_ctx=chat_ctx)

    @function_tool
    async def explain_concept(self, context: RunContext[TutorData], concept_id: str = None) -> str:
        """Explain a specific concept."""
        if concept_id:
            concept = context.userdata.concepts.get(concept_id)
            if not concept:
                return f"Concept '{concept_id}' not found."
        else:
            # Use current concept or ask for one
            concept = context.userdata.current_concept
            if not concept:
                available = ', '.join(c['title'] for c in context.userdata.concepts.values())
                return f"Which concept would you like me to explain? Available: {available}"
        
        context.userdata.current_concept = concept
        context.userdata.current_mode = 'learn'
        
        explanation = f"Let me explain {concept['title']}:\n\n{concept['summary']}\n\nDoes that make sense? Would you like me to elaborate on any part, or would you prefer to quiz yourself on this concept?"
        return explanation

    @function_tool
    async def transfer_to_quiz_mode(self, context: RunContext[TutorData]) -> Agent:
        """Switch to quiz mode."""
        await self.session.say("Alright, let's test your understanding! Switching to quiz mode.")
        return await self._transfer_to_agent("quiz", context)

    @function_tool
    async def transfer_to_teach_back_mode(self, context: RunContext[TutorData]) -> Agent:
        """Switch to teach back mode."""
        await self.session.say("Great! Now you can practice explaining this concept back to us.")
        return await self._transfer_to_agent("teach_back", context)


class QuizAgent(BaseAgent):
    """Agent for quizzing users on programming concepts."""
    
    def __init__(self, chat_ctx: Optional[ChatContext] = None) -> None:
        instructions = """You are Alicia, an engaging and questioning tutor focused on testing understanding through quizzes.

        CRITICAL: When you first take over from another agent, IMMEDIATELY start with this exact introduction:
        "Hi there! I'm Alicia, your quiz master! I love testing knowledge and helping you discover what you've learned. Which concept are you ready to be quizzed on?"

        Your role is to:
        - Always start with the introduction above when you first take control
        - Ask quiz questions about programming concepts using the sample questions from the content file
        - Make questions challenging but fair
        - Give hints if users struggle
        - Provide encouraging feedback on answers
        - Allow users to switch to other modes when they're ready

        Available concepts: variables, loops, functions, conditionals

        Be encouraging and supportive. When users want to switch modes, transfer them appropriately."""

        super().__init__(instructions=instructions, voice_key="quiz", chat_ctx=chat_ctx)

    @function_tool
    async def ask_quiz_question(self, context: RunContext[TutorData], concept_id: str = None) -> str:
        """Ask a quiz question about a concept."""
        if concept_id:
            concept = context.userdata.concepts.get(concept_id)
            if not concept:
                return f"Concept '{concept_id}' not found."
        else:
            concept = context.userdata.current_concept
            if not concept:
                available = ', '.join(c['title'] for c in context.userdata.concepts.values())
                return f"Which concept should I quiz you on? Available: {available}"
        
        context.userdata.current_concept = concept
        context.userdata.current_mode = 'quiz'
        
        return f"Here's a question about {concept['title']}: {concept['sample_question']}"

    @function_tool
    async def transfer_to_learn_mode(self, context: RunContext[TutorData]) -> Agent:
        """Switch to learn mode."""
        await self.session.say("Let's go back to learning mode for some explanations.")
        return await self._transfer_to_agent("learn", context)

    @function_tool
    async def transfer_to_teach_back_mode(self, context: RunContext[TutorData]) -> Agent:
        """Switch to teach back mode."""
        await self.session.say("Now let's see if you can explain this concept back to me!")
        return await self._transfer_to_agent("teach_back", context)


class TeachBackAgent(BaseAgent):
    """Agent for having users teach concepts back."""
    
    def __init__(self, chat_ctx: Optional[ChatContext] = None) -> None:
        instructions = """You are Ken, an encouraging and evaluative tutor focused on active recall through teaching back.

        CRITICAL: When you first take over from another agent, IMMEDIATELY start with this exact introduction:
        "Hello! I'm Ken, your teach-back coach! This is where the real learning magic happens. I'll listen as you explain concepts in your own words, and I'll give you helpful feedback to strengthen your understanding. What concept would you like to teach me about?"

        Your role is to:
        - Always start with the introduction above when you first take control
        - Ask users to explain programming concepts in their own words
        - Provide constructive feedback on their explanations
        - Score their understanding based on key concepts covered
        - Give specific suggestions for improvement
        - Allow users to switch to other modes when they're ready

        Available concepts: variables, loops, functions, conditionals

        Be encouraging and supportive. Focus on what they got right and gently suggest areas for improvement."""

        super().__init__(instructions=instructions, voice_key="teach_back", chat_ctx=chat_ctx)

    @function_tool
    async def prompt_teach_back(self, context: RunContext[TutorData], concept_id: str = None) -> str:
        """Prompt user to teach back a concept."""
        if concept_id:
            concept = context.userdata.concepts.get(concept_id)
            if not concept:
                return f"Concept '{concept_id}' not found."
        else:
            concept = context.userdata.current_concept
            if not concept:
                available = ', '.join(c['title'] for c in context.userdata.concepts.values())
                return f"Which concept would you like to teach me? Available: {available}"
        
        context.userdata.current_concept = concept
        context.userdata.current_mode = 'teach_back'
        
        return f"Please explain {concept['title']} back to me in your own words. Take your time and cover the key points you remember."

    @function_tool
    async def evaluate_explanation(self, context: RunContext[TutorData], user_explanation: str) -> str:
        """Evaluate the user's explanation and provide feedback."""
        concept = context.userdata.current_concept
        if not concept:
            return "Please select a concept first."
        
        # Simple evaluation based on key terms
        user_lower = user_explanation.lower()
        concept_terms = CONFIG["key_terms"].get(concept['id'], [])
        found_terms = [term for term in concept_terms if term in user_lower]
        
        # Provide feedback
        if len(found_terms) >= len(concept_terms) * 0.6:
            feedback = f"Excellent work! You covered the key aspects of {concept['title']}. "
            if len(found_terms) == len(concept_terms):
                feedback += "You mentioned all the important concepts!"
            else:
                feedback += f"You got most of the important points. Great understanding!"
        elif len(found_terms) >= len(concept_terms) * 0.3:
            feedback = f"Good start! You understand some key aspects of {concept['title']}. "
            missing = len(concept_terms) - len(found_terms)
            feedback += f"Consider also mentioning aspects like how {concept['id']} help with organization and reusability in programming."
        else:
            feedback = f"I can see you're working on understanding {concept['title']}. "
            feedback += f"Let me help clarify: {concept['summary'][:100]}... Would you like to try explaining it again?"
        
        feedback += f"\n\nWould you like to switch to a different mode or concept, or continue practicing?"
        return feedback

    @function_tool
    async def transfer_to_learn_mode(self, context: RunContext[TutorData]) -> Agent:
        """Switch to learn mode."""
        await self.session.say("Let's review the concepts in learn mode.")
        return await self._transfer_to_agent("learn", context)

    @function_tool
    async def transfer_to_quiz_mode(self, context: RunContext[TutorData]) -> Agent:
        """Switch to quiz mode."""
        await self.session.say("Ready for a quiz to test your knowledge?")
        return await self._transfer_to_agent("quiz", context)


def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


async def entrypoint(ctx: JobContext):
    # Logging setup
    ctx.log_context_fields = {
        "room": ctx.room.name,
    }

    # Initialize shared data
    userdata = TutorData()
    
    # Create all agents
    tutor_agent = TutorAgent()
    learn_agent = LearnAgent()
    quiz_agent = QuizAgent()
    teach_back_agent = TeachBackAgent()
    
    # Load concepts into userdata
    content_file = CONFIG["content_file"]
    try:
        if content_file.exists():
            with open(content_file, 'r', encoding='utf-8') as f:
                concepts_list = json.load(f)
                userdata.concepts = {c['id']: c for c in concepts_list}
                logger.info(f"Loaded {len(userdata.concepts)} programming concepts")
        else:
            logger.error(f"Content file not found: {content_file}")
            userdata.concepts = {}
    except Exception as e:
        logger.error(f"Error loading content: {e}")
        userdata.concepts = {}
    
    # Register agents for handoffs
    userdata.personas = {
        "tutor": tutor_agent,
        "learn": learn_agent,
        "quiz": quiz_agent,
        "teach_back": teach_back_agent
    }

    # Create the session with the main tutor agent
    session = AgentSession[TutorData](
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

    # Start the session with the main tutor agent
    await session.start(
        agent=tutor_agent,
        room=ctx.room,
        room_input_options=RoomInputOptions(
            noise_cancellation=noise_cancellation.BVC(),
        ),
    )

    # Join the room and connect to the user
    await ctx.connect()


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))