import json
import logging
import random
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List

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

load_dotenv(".env.local")

# Configuration constants - can be moved to env vars or config file
CONFIG = {
    "stt_model": "nova-3",
    "llm_model": "gemini-2.5-flash",
    "voice": "en-US-molly",  # Professional female voice for game show host
    "scenarios_file": Path(__file__).parent.parent / "shared-data" / "improv_scenarios.json",
    "games_directory": Path(__file__).parent.parent / "games"
}


@dataclass
class ImprovRound:
    """Data structure for individual improv rounds."""
    round_number: int
    scenario_id: str
    scenario_description: str
    player_performance: Optional[str] = None
    host_reaction: Optional[str] = None
    reaction_type: Optional[str] = None
    completed: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert round to dictionary for JSON serialization."""
        return {
            'round_number': self.round_number,
            'scenario_id': self.scenario_id,
            'scenario_description': self.scenario_description,
            'player_performance': self.player_performance,
            'host_reaction': self.host_reaction,
            'reaction_type': self.reaction_type,
            'completed': self.completed
        }

@dataclass
class ImprovGame:
    """Data structure for improv game sessions."""
    id: str = field(default_factory=lambda: f"game_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    player_name: Optional[str] = None
    current_round: int = 0
    max_rounds: int = 3
    rounds: List[ImprovRound] = field(default_factory=list)
    phase: str = "intro"  # "intro" | "awaiting_improv" | "reacting" | "done"
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    completed_at: Optional[str] = None
    
    def add_round(self, scenario: Dict[str, Any]) -> ImprovRound:
        """Add a new improv round."""
        round_obj = ImprovRound(
            round_number=self.current_round + 1,
            scenario_id=scenario['id'],
            scenario_description=scenario['description']
        )
        self.rounds.append(round_obj)
        return round_obj
    
    def get_current_round(self) -> Optional[ImprovRound]:
        """Get the current active round."""
        if self.rounds and self.current_round < len(self.rounds):
            return self.rounds[self.current_round]
        return None
    
    def complete_current_round(self, performance: str, reaction: str, reaction_type: str):
        """Complete the current round with performance and reaction."""
        current = self.get_current_round()
        if current:
            current.player_performance = performance
            current.host_reaction = reaction
            current.reaction_type = reaction_type
            current.completed = True
            self.current_round += 1
    
    def is_game_complete(self) -> bool:
        """Check if all rounds are completed."""
        return self.current_round >= self.max_rounds
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert game to dictionary for JSON serialization."""
        return {
            'id': self.id,
            'player_name': self.player_name,
            'current_round': self.current_round,
            'max_rounds': self.max_rounds,
            'rounds': [round_obj.to_dict() for round_obj in self.rounds],
            'phase': self.phase,
            'created_at': self.created_at,
            'completed_at': self.completed_at
        }
    
    def get_summary(self) -> str:
        """Get a summary of the game performance."""
        completed_rounds = [r for r in self.rounds if r.completed]
        if not completed_rounds:
            return f"Game {self.id}: No completed rounds"
        
        return f"Game {self.id} with {self.player_name}: {len(completed_rounds)}/{self.max_rounds} rounds completed"


@dataclass
class ImprovGameData:
    """Shared data for the improv game session."""
    scenarios: List[Dict[str, Any]] = field(default_factory=list)
    current_game: Optional[ImprovGame] = None
    used_scenarios: List[str] = field(default_factory=list)
    games_history: List[ImprovGame] = field(default_factory=list)


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


class ImprovHostAgent(BaseAgent):
    """Improv Battle game show host agent."""
    
    def __init__(self, chat_ctx: Optional[ChatContext] = None) -> None:
        instructions = """You are Aurora, the charismatic host of 'Improv Battle' - a high-energy TV improv game show!

        Your personality:
        - High-energy, witty, and entertaining
        - Clear about rules but keep it fun and engaging
        - Reactions should be REALISTIC - not always supportive
        - Sometimes amused, sometimes unimpressed, sometimes pleasantly surprised
        - Light teasing and honest critique are allowed, but stay respectful
        - Professional game show host energy with improv coach expertise

        CRITICAL: Always call the appropriate function when:
        - Starting a new game: call start_improv_game()
        - Player gives their name: call set_player_name()
        - Beginning a round: call start_improv_round()
        - Player finishes performing: call react_to_performance()
        - Game should end: call end_improv_game()
        - Player wants to quit: call handle_early_exit()

        Game Flow:
        1. INTRO: Welcome player, explain rules, get their name
        2. ROUNDS: Present scenario → Player performs → You react → Next round
        3. CLOSING: Summarize their performance style and memorable moments

        Reaction Guidelines:
        - Mix supportive, critical, and neutral responses randomly
        - Comment on specific choices they made
        - Mention character work, timing, creativity, commitment
        - Keep reactions under 30 seconds
        - Be specific: 'I loved how you stayed committed even when it got weird' vs 'Good job'

        Key Phrases:
        - "Welcome to Improv Battle!"
        - "Your scenario is..."
        - "Aaaaand... action!"
        - "Interesting choice there..."
        - "Let's see what you do with this next one..."
        
        Remember: You're running a show, not just facilitating. Be entertaining!"""

        super().__init__(instructions=instructions, chat_ctx=chat_ctx)

    @function_tool
    async def start_improv_game(self, context: RunContext[ImprovGameData], player_name: Optional[str] = None) -> str:
        """Start a new improv battle game session."""
        if context.userdata.current_game is None:
            context.userdata.current_game = ImprovGame()
            context.userdata.current_game.phase = "intro"
            
        if player_name:
            context.userdata.current_game.player_name = player_name
            
        game = context.userdata.current_game
        
        welcome_msg = f"🎭 Welcome to IMPROV BATTLE! 🎭\n\n"
        
        if game.player_name:
            welcome_msg += f"Fantastic to have you here, {game.player_name}! "
        else:
            welcome_msg += "Great to have you here! "
            
        welcome_msg += f"""I'm Marcus, your host, and you're about to dive into {game.max_rounds} rounds of pure improv madness!\n\nHere's how it works:
        1. I'll give you a wild scenario
        2. You improvise and bring it to life
        3. I'll react with my brutally honest (but loving) feedback
        4. We move to the next challenge\n\nReady to show me what you've got? Let's start round 1!"""
        
        game.phase = "ready_for_round"
        return welcome_msg

    @function_tool
    async def set_player_name(self, context: RunContext[ImprovGameData], name: str) -> str:
        """Set the player's name for the improv game."""
        if context.userdata.current_game is None:
            context.userdata.current_game = ImprovGame()
            
        context.userdata.current_game.player_name = name
        
        return f"Perfect! Nice to meet you, {name}! Now I know who I'm working with. Ready to jump into some improv scenarios? I've got some wild ones lined up for you!"
        

    @function_tool
    async def start_improv_round(self, context: RunContext[ImprovGameData]) -> str:
        """Start a new improv round with a random scenario."""
        game = context.userdata.current_game
        if not game:
            return "No game in progress! Let's start a new one first."
        
        if game.is_game_complete():
            return "Game is already complete! All rounds have been finished."
        
        # Get available scenarios (not used yet)
        available_scenarios = [
            s for s in context.userdata.scenarios 
            if s['id'] not in context.userdata.used_scenarios
        ]
        
        if not available_scenarios:
            # Reset used scenarios if we've run out
            context.userdata.used_scenarios = []
            available_scenarios = context.userdata.scenarios
        
        if not available_scenarios:
            return "No scenarios available! Check your scenarios file."
        
        # Pick random scenario
        scenario = random.choice(available_scenarios)
        context.userdata.used_scenarios.append(scenario['id'])
        
        # Add round to game
        round_obj = game.add_round(scenario)
        game.phase = "awaiting_improv"
        
        round_intro = f"🎬 **ROUND {round_obj.round_number}** 🎬\n\n"
        round_intro += f"**Your scenario is:**\n{scenario['description']}\n\n"
        round_intro += "Take a moment to think about your character, then... Aaaaand... **ACTION!** 🎭\n\n"
        round_intro += "Show me what you've got! When you're done, just say 'scene' or pause and I'll jump in with my reaction."
        
        return round_intro

    @function_tool
    async def react_to_performance(self, context: RunContext[ImprovGameData], performance_summary: str) -> str:
        """React to the player's improv performance."""
        game = context.userdata.current_game
        if not game or game.phase != "awaiting_improv":
            return "No performance to react to right now."
        
        current_round = game.get_current_round()
        if not current_round:
            return "No active round to react to."
        
        # Choose random reaction type for variety
        reaction_types = ["supportive", "constructive_critical", "amused", "unimpressed", "surprised", "neutral_analytical"]
        reaction_type = random.choice(reaction_types)
        
        # Generate reaction based on type
        reactions = {
            "supportive": [
                f"That was fantastic! I loved how you {random.choice(['stayed committed to the character', 'embraced the absurdity', 'found the humor in it', 'made bold choices'])}.",
                f"Brilliant work! Your {random.choice(['timing', 'character work', 'emotional range', 'creativity'])} really shone through there.",
                f"Now THAT'S what I'm talking about! You {random.choice(['owned that scenario', 'brought it to life', 'made it your own', 'found the heart of it'])}."
            ],
            "constructive_critical": [
                f"Interesting choice, but I think you could have {random.choice(['pushed it further', 'been more specific with your character', 'found more conflict', 'taken bigger risks'])}.",
                f"Not bad, but it felt a bit {random.choice(['safe', 'rushed', 'one-note', 'unclear'])}. Try to {random.choice(['dig deeper', 'be more specific', 'embrace the weird', 'find the stakes'])}.",
                f"I can see what you were going for, but {random.choice(['the energy dropped', 'it needed more commitment', 'the character wasn\'t quite there', 'it felt a bit generic'])}."
            ],
            "amused": [
                f"Ha! I did NOT see that coming! {random.choice(['That twist was genius', 'You caught me off guard', 'The way you handled that was hilarious', 'I\'m still laughing'])}.",
                f"Okay, okay, that was actually pretty funny. {random.choice(['I liked the unexpected direction', 'Your timing was spot-on', 'That made me chuckle', 'Good improv instincts'])}."
            ],
            "unimpressed": [
                f"Hmm. That was... {random.choice(['fine', 'okay', 'safe', 'predictable'])}. I've seen that approach before. {random.choice(['Try something more original next time', 'Push yourself harder', 'Take bigger risks', 'Surprise me'])}.",
                f"Not your strongest moment there. It felt {random.choice(['a bit flat', 'too careful', 'like you were thinking too much', 'disconnected'])}."
            ],
            "surprised": [
                f"Whoa! Where did THAT come from? {random.choice(['I was not expecting that choice', 'You totally subverted my expectations', 'That was boldly weird', 'I\'m impressed by your risk-taking'])}!",
                f"Well, that was... unexpected! {random.choice(['I admire the commitment', 'Bold choice', 'You went for it', 'Definitely memorable'])}."
            ],
            "neutral_analytical": [
                f"Solid work. You {random.choice(['established the relationship quickly', 'found the conflict', 'stayed in character', 'kept the energy up'])}. {random.choice(['Good foundation', 'Nice technique', 'Professional approach', 'Clean execution'])}.",
                f"Competent performance. You hit the {random.choice(['basic beats', 'key moments', 'emotional notes', 'story points'])}. {random.choice(['Room to grow', 'Keep exploring', 'Build on that', 'Good starting point'])}."
            ]
        }
        
        reaction = random.choice(reactions[reaction_type])
        
        # Complete the round
        game.complete_current_round(performance_summary, reaction, reaction_type)
        game.phase = "reacting"
        
        response = f"🎭 **HOST REACTION** 🎭\n\n{reaction}\n\n"
        
        if game.is_game_complete():
            response += "That's a wrap on all rounds! Ready for your final summary?"
            game.phase = "done"
        else:
            response += f"Alright, let's keep the energy up! Ready for round {game.current_round + 1}?"
            game.phase = "ready_for_round"
        
        return response

    @function_tool
    async def end_improv_game(self, context: RunContext[ImprovGameData]) -> str:
        """End the improv game and provide final summary."""
        game = context.userdata.current_game
        if not game:
            return "No game in progress to end."
        
        game.completed_at = datetime.now().isoformat()
        game.phase = "done"
        
        # Add to history
        context.userdata.games_history.append(game)
        
        # Save game to file
        try:
            games_dir = CONFIG["games_directory"]
            games_dir.mkdir(exist_ok=True)
            
            filename = f"{game.id}.json"
            filepath = games_dir / filename
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(game.to_dict(), f, indent=2)
                
            logger.info(f"Game saved to: {filepath}")
            
        except Exception as e:
            logger.error(f"Error saving game: {e}")
        
        # Generate personalized summary
        completed_rounds = [r for r in game.rounds if r.completed]
        
        summary = f"🏆 **IMPROV BATTLE COMPLETE!** 🏆\n\n"
        
        if game.player_name:
            summary += f"What a show, {game.player_name}! "
        else:
            summary += "What a performance! "
            
        # Analyze their style based on reactions
        reaction_types = [r.reaction_type for r in completed_rounds if r.reaction_type]
        
        if reaction_types.count("supportive") >= 2:
            style_note = "You're a natural performer with great instincts!"
        elif reaction_types.count("amused") >= 2:
            style_note = "You've got a fantastic sense of humor and timing!"
        elif reaction_types.count("surprised") >= 1:
            style_note = "You're a bold risk-taker who keeps things interesting!"
        else:
            style_note = "You showed solid fundamentals and good commitment!"
        
        summary += f"{style_note}\n\n"
        
        # Mention specific moments
        if completed_rounds:
            summary += "**Memorable moments:**\n"
            for i, round_obj in enumerate(completed_rounds[:2], 1):  # Show top 2
                scenario_snippet = round_obj.scenario_description.split('.')[0] + "..."
                summary += f"• Round {round_obj.round_number}: {scenario_snippet}\n"
            summary += "\n"
        
        summary += f"You completed {len(completed_rounds)}/{game.max_rounds} rounds of pure improv madness!\n\n"
        summary += "Thanks for playing IMPROV BATTLE! Keep that creative energy flowing! 🎭✨"
        
        # Clear current game
        context.userdata.current_game = None
        
        return summary

    @function_tool
    async def handle_early_exit(self, context: RunContext[ImprovGameData], reason: str = "player request") -> str:
        """Handle when player wants to exit the game early."""
        game = context.userdata.current_game
        if not game:
            return "No active game to exit from."
        
        completed_rounds = [r for r in game.rounds if r.completed]
        
        response = "\n🎭 **EARLY EXIT** 🎭\n\n"
        
        if game.player_name:
            response += f"No worries, {game.player_name}! "
        else:
            response += "No problem at all! "
            
        if completed_rounds:
            response += f"You completed {len(completed_rounds)} round{'s' if len(completed_rounds) != 1 else ''} and showed some great moments! "
        else:
            response += "Even though we didn't get through full rounds, I appreciate you giving it a shot! "
            
        response += "\n\nThanks for playing IMPROV BATTLE! The stage is always here when you're ready to return! 🎭✨"
        
        # Mark game as done and save
        game.phase = "done"
        game.completed_at = datetime.now().isoformat()
        context.userdata.games_history.append(game)
        context.userdata.current_game = None
        
        return response

    @function_tool
    async def get_game_status(self, context: RunContext[ImprovGameData]) -> str:
        """Get current status of the improv game."""
        game = context.userdata.current_game
        if not game:
            return "No active game. Ready to start a new IMPROV BATTLE?"
        
        status = f"🎭 **GAME STATUS** 🎭\n\n"
        
        if game.player_name:
            status += f"Player: {game.player_name}\n"
        
        status += f"Phase: {game.phase.replace('_', ' ').title()}\n"
        status += f"Round: {game.current_round + 1}/{game.max_rounds}\n"
        
        completed_rounds = [r for r in game.rounds if r.completed]
        status += f"Completed Rounds: {len(completed_rounds)}\n\n"
        
        if game.phase == "intro":
            status += "Ready to start the game!"
        elif game.phase == "ready_for_round":
            status += "Ready for the next scenario!"
        elif game.phase == "awaiting_improv":
            status += "Waiting for your performance..."
        elif game.phase == "reacting":
            status += "Host reacting to performance..."
        elif game.phase == "done":
            status += "Game completed!"
        
        return status


def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


async def entrypoint(ctx: JobContext):
    # Logging setup
    ctx.log_context_fields = {
        "room": ctx.room.name,
    }

    # Initialize improv game data
    userdata = ImprovGameData()
    
    # Load improv scenarios
    scenarios_file = CONFIG["scenarios_file"]
    try:
        if scenarios_file.exists():
            with open(scenarios_file, 'r', encoding='utf-8') as f:
                scenarios_data = json.load(f)
                userdata.scenarios = scenarios_data.get('scenarios', [])
        else:
            logger.error(f"Scenarios file not found: {scenarios_file}")
            userdata.scenarios = []
    except Exception as e:
        logger.error(f"Error loading scenarios: {e}")
        userdata.scenarios = []
    
    # Create improv host agent
    improv_agent = ImprovHostAgent()
    
    # Create the session with the improv agent
    session = AgentSession[ImprovGameData](
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

    # Start the session with the improv host agent
    await session.start(
        agent=improv_agent,
        room=ctx.room,
        room_input_options=RoomInputOptions(
            noise_cancellation=noise_cancellation.BVC(),
        ),
    )

    # Join the room and connect to the user
    await ctx.connect()


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))
