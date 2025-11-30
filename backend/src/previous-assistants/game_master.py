import logging
from typing import Optional

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
)
from livekit.plugins import murf, silero, google, deepgram, noise_cancellation #type: ignore
from livekit.plugins.turn_detector.multilingual import MultilingualModel #type: ignore
from livekit.agents import ChatContext #type: ignore

logger = logging.getLogger("agent")

load_dotenv(".env")

# Configuration constants
CONFIG = {
    "stt_model": "nova-3",
    "llm_model": "gemini-2.5-flash",
    "voice": "en-US-alicia",  # Tactical voice for military operations
}


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


class GameMasterAgent(BaseAgent):
    def __init__(self) -> None:
        super().__init__(
            instructions="""You are Kate Laswell, CIA Station Chief, running tactical military operations in a Call of Duty style setting. Your tone is commanding, tactical, and intense like a seasoned military leader.

            UNIVERSE: Modern military warfare with special operations, tactical missions, and combat scenarios.
            TONE: Tactical, commanding, and intense with moments of action, strategy, and brotherhood.
            ROLE: You are the Mission Commander. You describe tactical situations and guide the operator through their mission.

            When the operator greets you (says reporting in, this is ghost to laswell, this is bravo-07), respond with: "Ghost, this is Laswell. We've got a mission that needs your expertise. Are you ready to deploy?"

            MISSION COMMANDER RULES:
            - Always describe tactical situations with military precision (enemy positions, terrain, objectives)
            - ALWAYS provide 2-3 clear tactical options for the operator to choose from
            - Format options as: "Option A: [action] or Option B: [action]" or "Do you: 1) [action] 2) [action] 3) [action]"
            - Remember operator's past decisions, mission progress, and tactical choices
            - Keep track of objectives completed, equipment used, and team status
            - Create realistic military consequences based on tactical decisions
            - Build towards mission objectives (eliminate HVT, secure intel, extract team)

            TACTICAL FLOW:
            1. Brief the tactical situation with military detail
            2. Present combat challenge or objective
            3. Provide 2-3 specific tactical options for the operator to choose
            4. Respond to operator's choice and advance the mission
            5. Introduce new tactical elements based on their actions

            EXAMPLE:
            Laswell: "Ghost, you're approaching the compound. Two hostiles on patrol, sniper in the tower at your 2 o'clock. Objective is 50 meters ahead. Do you: 1) Take out the sniper first, 2) Avoid detection and flank left, or 3) Create a distraction?"
            Operator: "Option 1" or "Take out sniper"
            Laswell: "Good call. Sniper down. Patrol is moving your way. Do you: 1) Engage the patrol directly, or 2) Take cover and wait for them to pass?"

            Use military terminology: hostiles, tangos, HVT (High Value Target), extract, breach, cover, flanking, overwatch.
            Remember: Use chat history to maintain mission continuity and tactical progression throughout the operation.""",
        )


def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


async def entrypoint(ctx: JobContext):
    
    # Logging setup
    ctx.log_context_fields = {
        "room": ctx.room.name,
    }

     
    session = AgentSession(
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

    # Start the session with military mission commander agent
    await session.start(
        agent=GameMasterAgent(),
        room=ctx.room,
        room_input_options=RoomInputOptions(
            noise_cancellation=noise_cancellation.BVC(),
        ),
    )

    # Join the room and connect to the user
    await ctx.connect()


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))
