import logging
import json
# import os
from datetime import datetime
from pathlib import Path

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

logger = logging.getLogger("agent")

load_dotenv(".env")


class TutorAssistant(Agent):
    def __init__(self) -> None:
        # Base instructions for the tutoring system
        base_instructions = """You are an Active Recall Coach that helps users learn programming concepts through three distinct learning modes:

LEARN mode: You explain concepts clearly and simply. Your voice is Matthew (warm, instructional).
QUIZ mode: You ask questions to test understanding. Your voice is Alicia (engaging, questioning).
TEACH_BACK mode: You ask the user to explain concepts back to you and give feedback. Your voice is Ken (encouraging, evaluative).

Always start by greeting the user and asking which learning mode they'd like to use. Users can switch modes at any time by asking.

Key behaviors:
- Keep explanations clear and beginner-friendly
- Ask follow-up questions to deepen understanding  
- Give constructive feedback when users teach back
- Use the available concepts from the content file
- Be encouraging and supportive throughout"""
        
        super().__init__(instructions=base_instructions)
        
        # Initialize learning state
        self.current_mode = None  # 'learn', 'quiz', 'teach_back'
        self.current_concept = None
        self.concepts = []
        
        # Load learning content
        self.content_file = Path(__file__).parent.parent / "shared-data" / "day4_tutor_content.json"
        self.load_content()

    def load_content(self):
        """Load programming concepts from the JSON file."""
        try:
            if self.content_file.exists():
                with open(self.content_file, 'r', encoding='utf-8') as f:
                    self.concepts = json.load(f)
                    logger.info(f"Loaded {len(self.concepts)} programming concepts")
            else:
                logger.error(f"Content file not found: {self.content_file}")
                self.concepts = []
        except Exception as e:
            logger.error(f"Error loading content: {e}")
            self.concepts = []

    @function_tool
    async def set_learning_mode(self, context: RunContext, mode: str, concept_id: str = None):
        """Set the learning mode and optionally select a concept.
        
        Args:
            mode: Learning mode - 'learn', 'quiz', or 'teach_back'
            concept_id: Optional concept ID to focus on (variables, loops, functions, conditionals)
        """
        valid_modes = ['learn', 'quiz', 'teach_back']
        if mode not in valid_modes:
            return f"Invalid mode. Please choose from: {', '.join(valid_modes)}"
        
        self.current_mode = mode
        
        # If concept_id provided, validate and set it
        if concept_id:
            concept = next((c for c in self.concepts if c['id'] == concept_id), None)
            if concept:
                self.current_concept = concept
            else:
                return f"Concept '{concept_id}' not found. Available concepts: {', '.join(c['id'] for c in self.concepts)}"
        
        # Mode-specific responses with voice confirmation
        if mode == 'learn':
            if self.current_concept:
                return f"Great! I'm now in LEARN mode and you'll hear Matthew's warm, instructional voice. Let me explain {self.current_concept['title']} to you."
            else:
                concept_list = ', '.join(c['title'] for c in self.concepts)
                return f"I'm now in LEARN mode and you'll hear Matthew's warm, instructional voice. Which concept would you like to learn about? Available: {concept_list}"
        
        elif mode == 'quiz':
            if self.current_concept:
                return f"Excellent! I'm now in QUIZ mode and you'll hear Alicia's engaging, questioning voice. Let me ask you about {self.current_concept['title']}."
            else:
                concept_list = ', '.join(c['title'] for c in self.concepts)
                return f"I'm now in QUIZ mode and you'll hear Alicia's engaging, questioning voice. Which concept should I quiz you on? Available: {concept_list}"
        
        elif mode == 'teach_back':
            if self.current_concept:
                return f"Perfect! I'm now in TEACH_BACK mode and you'll hear Ken's encouraging, evaluative voice. Please explain {self.current_concept['title']} back to me."
            else:
                concept_list = ', '.join(c['title'] for c in self.concepts)
                return f"I'm now in TEACH_BACK mode and you'll hear Ken's encouraging, evaluative voice. Which concept would you like to teach me? Available: {concept_list}"

    @function_tool
    async def select_concept(self, context: RunContext, concept_id: str):
        """Select a specific concept to work with.
        
        Args:
            concept_id: The concept ID (variables, loops, functions, conditionals)
        """
        concept = next((c for c in self.concepts if c['id'] == concept_id), None)
        if not concept:
            available = ', '.join(c['id'] for c in self.concepts)
            return f"Concept '{concept_id}' not found. Available concepts: {available}"
        
        self.current_concept = concept
        
        if not self.current_mode:
            return f"Great! I've selected {concept['title']}. Which learning mode would you like to use? (learn, quiz, or teach_back)"
        
        # Provide mode-specific response with the new concept
        if self.current_mode == 'learn':
            return f"Perfect! Let me explain {concept['title']} to you: {concept['summary']}"
        elif self.current_mode == 'quiz':
            return f"Great! Here's a question about {concept['title']}: {concept['sample_question']}"
        elif self.current_mode == 'teach_back':
            return f"Excellent! Now please explain {concept['title']} back to me in your own words."

    @function_tool
    async def explain_concept(self, context: RunContext, concept_id: str = None):
        """Explain a programming concept in LEARN mode.
        
        Args:
            concept_id: Optional concept ID to explain, uses current concept if not provided
        """
        if concept_id:
            concept = next((c for c in self.concepts if c['id'] == concept_id), None)
            if not concept:
                return f"Concept '{concept_id}' not found."
        else:
            concept = self.current_concept
        
        if not concept:
            available = ', '.join(c['title'] for c in self.concepts)
            return f"Please select a concept first. Available: {available}"
        
        # Set to learn mode and current concept
        self.current_mode = 'learn'
        self.current_concept = concept
        
        explanation = f"Let me explain {concept['title']}:\n\n{concept['summary']}\n\nWould you like me to quiz you on this concept, or would you prefer to teach it back to me to test your understanding?"
        
        return explanation

    @function_tool
    async def ask_quiz_question(self, context: RunContext, concept_id: str = None):
        """Ask a quiz question in QUIZ mode.
        
        Args:
            concept_id: Optional concept ID to quiz on, uses current concept if not provided
        """
        if concept_id:
            concept = next((c for c in self.concepts if c['id'] == concept_id), None)
            if not concept:
                return f"Concept '{concept_id}' not found."
        else:
            concept = self.current_concept
        
        if not concept:
            available = ', '.join(c['title'] for c in self.concepts)
            return f"Please select a concept first. Available: {available}"
        
        # Set to quiz mode and current concept
        self.current_mode = 'quiz'
        self.current_concept = concept
        
        return f"Here's a question about {concept['title']}: {concept['sample_question']}"

    @function_tool
    async def evaluate_teaching(self, context: RunContext, user_explanation: str):
        """Evaluate the user's explanation in TEACH_BACK mode and provide feedback.
        
        Args:
            user_explanation: The user's explanation of the concept
        """
        if not self.current_concept:
            return "Please select a concept first before teaching back."
        
        # Set to teach_back mode
        self.current_mode = 'teach_back'
        
        # Simple evaluation based on key terms and concepts
        concept = self.current_concept
        concept_id = concept['id']
        
        # Key terms to look for in explanations
        key_terms = {
            'variables': ['store', 'value', 'container', 'label', 'reuse'],
            'loops': ['repeat', 'iterate', 'for', 'while', 'condition'],
            'functions': ['reusable', 'parameter', 'input', 'return', 'block'],
            'conditionals': ['if', 'condition', 'decision', 'true', 'false', 'else']
        }
        
        user_lower = user_explanation.lower()
        concept_terms = key_terms.get(concept_id, [])
        found_terms = [term for term in concept_terms if term in user_lower]
        
        # Provide feedback based on coverage
        if len(found_terms) >= len(concept_terms) * 0.6:  # 60% or more key terms
            feedback = f"Excellent explanation! You covered the key aspects of {concept['title']}. "
            if len(found_terms) == len(concept_terms):
                feedback += "You mentioned all the important concepts!"
            else:
                feedback += f"You got most of the important points. Great understanding!"
        elif len(found_terms) >= len(concept_terms) * 0.3:  # 30-60% key terms
            feedback = f"Good start! You understand some key aspects of {concept['title']}. "
            missing = len(concept_terms) - len(found_terms)
            feedback += f"Consider also mentioning aspects like how {concept_id} help with organization and reusability in programming."
        else:
            feedback = f"I can see you're working on understanding {concept['title']}. "
            feedback += f"Let me help clarify: {concept['summary'][:100]}... Would you like to try explaining it again?"
        
        feedback += f"\n\nWould you like to switch to a different mode or concept, or continue practicing?"
        
        return feedback

    @function_tool
    async def get_available_concepts(self, context: RunContext):
        """Get a list of all available programming concepts."""
        if not self.concepts:
            return "No concepts are currently loaded."
        
        concept_list = []
        for concept in self.concepts:
            concept_list.append(f"- {concept['title']} (ID: {concept['id']}): {concept['summary'][:50]}...")
        
        return "Available programming concepts:\n" + "\n".join(concept_list)

    @function_tool
    async def get_current_status(self, context: RunContext):
        """Get the current learning mode and concept status."""
        status = []
        
        if self.current_mode:
            voice_info = {
                'learn': 'Matthew (warm, instructional)',
                'quiz': 'Alicia (engaging, questioning)', 
                'teach_back': 'Ken (encouraging, evaluative)'
            }
            current_voice = voice_info.get(self.current_mode, 'Unknown')
            status.append(f"Current mode: {self.current_mode.upper()} with {current_voice} voice")
        else:
            status.append("No mode selected yet")
        
        if self.current_concept:
            status.append(f"Current concept: {self.current_concept['title']}")
        else:
            status.append("No concept selected")
        
        status.append("\nAvailable modes: learn (Matthew), quiz (Alicia), teach_back (Ken)")
        status.append(f"Available concepts: {', '.join(c['title'] for c in self.concepts)}")
        
        return "\n".join(status)


def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


async def entrypoint(ctx: JobContext):
    # Logging setup
    ctx.log_context_fields = {
        "room": ctx.room.name,
    }

    # Create the tutor assistant
    tutor = TutorAssistant()
    
    # Pre-create TTS instances for each voice
    tts_voices = {
        'learn': murf.TTS(
            voice="en-US-matthew",  # Matthew - warm, instructional
            style="Conversation",
            tokenizer=tokenize.basic.SentenceTokenizer(min_sentence_len=2),
            text_pacing=True
        ),
        'quiz': murf.TTS(
            voice="en-US-alicia",   # Alicia - engaging, questioning
            style="Conversation", 
            tokenizer=tokenize.basic.SentenceTokenizer(min_sentence_len=2),
            text_pacing=True
        ),
        'teach_back': murf.TTS(
            voice="en-US-ken",      # Ken - encouraging, evaluative
            style="Conversation",
            tokenizer=tokenize.basic.SentenceTokenizer(min_sentence_len=2), 
            text_pacing=True
        )
    }
    
    # Set up a voice AI pipeline - start with Matthew (learn mode)
    session = AgentSession(
        # Speech-to-text (STT)
        stt=deepgram.STT(model="nova-3"),
        # Large Language Model (LLM) 
        llm=google.LLM(model="gemini-2.5-flash"),
        # Text-to-speech (TTS) - start with Matthew voice for learn mode
        tts=tts_voices['learn'],
        # VAD and turn detection
        turn_detection=MultilingualModel(),
        vad=ctx.proc.userdata["vad"],
        preemptive_generation=True,
    )

    # Function to switch TTS voice based on mode
    async def switch_voice_for_mode(mode):
        if mode in tts_voices:
            session._tts = tts_voices[mode]
            logger.info(f"Switched to voice for {mode} mode")
            return True
        return False

    # Enhanced set_learning_mode function that also switches voice
    original_set_learning_mode = tutor.set_learning_mode
    async def enhanced_set_learning_mode(context, mode, concept_id=None):
        # Switch voice first
        await switch_voice_for_mode(mode)
        # Then call original function
        return await original_set_learning_mode(context, mode, concept_id)
    
    # Replace the original method
    tutor.set_learning_mode = enhanced_set_learning_mode

    # Also enhance other mode-switching functions
    original_explain_concept = tutor.explain_concept
    async def enhanced_explain_concept(context, concept_id=None):
        await switch_voice_for_mode('learn')
        return await original_explain_concept(context, concept_id)
    tutor.explain_concept = enhanced_explain_concept

    original_ask_quiz_question = tutor.ask_quiz_question  
    async def enhanced_ask_quiz_question(context, concept_id=None):
        await switch_voice_for_mode('quiz')
        return await original_ask_quiz_question(context, concept_id)
    tutor.ask_quiz_question = enhanced_ask_quiz_question

    original_evaluate_teaching = tutor.evaluate_teaching
    async def enhanced_evaluate_teaching(context, user_explanation):
        await switch_voice_for_mode('teach_back')
        return await original_evaluate_teaching(context, user_explanation)
    tutor.evaluate_teaching = enhanced_evaluate_teaching

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

    # Start the session
    await session.start(
        agent=tutor,
        room=ctx.room,
        room_input_options=RoomInputOptions(
            noise_cancellation=noise_cancellation.BVC(),
        ),
    )

    # Join the room and connect to the user
    await ctx.connect()


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))
