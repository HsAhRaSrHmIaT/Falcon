import logging
import json
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
from uuid import uuid4

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


def load_food_catalog():
    """Load the food catalog from JSON file."""
    catalog_file = Path(__file__).parent.parent / "shared-data" / "food_catalog.json"
    
    try:
        with open(catalog_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading food catalog: {e}")
        return {"catalog": {"categories": {}}, "recipes": {}}


def save_order_to_file(order: Dict[str, Any]) -> str:
    """Save completed order to JSON file."""
    orders_dir = Path(__file__).parent.parent / "orders"
    orders_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    order_filename = f"order_FRESHMART_{timestamp}.json"
    order_file = orders_dir / order_filename
    
    try:
        with open(order_file, 'w', encoding='utf-8') as f:
            json.dump(order, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Order saved to {order_file}")
        return str(order_file)
    except Exception as e:
        logger.error(f"Error saving order: {e}")
        raise


# Configuration constants
CONFIG = {
    "stt_model": "nova-3",
    "llm_model": "gemini-2.5-flash",
    "voice": "en-US-alicia",  # Friendly voice for food ordering
}


@dataclass
class CartItem:
    """Represents an item in the shopping cart."""
    item_id: str
    name: str
    price: float
    quantity: int = 1
    category: str = ""
    brand: str = ""
    size: str = ""
    notes: str = ""
    
    def total_price(self) -> float:
        """Calculate total price for this cart item."""
        return self.price * self.quantity
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert cart item to dictionary."""
        return {
            'item_id': self.item_id,
            'name': self.name,
            'price': self.price,
            'quantity': self.quantity,
            'category': self.category,
            'brand': self.brand,
            'size': self.size,
            'notes': self.notes,
            'total_price': self.total_price()
        }


@dataclass
class OrderData:
    """Shared data for the food ordering session."""
    cart: List[CartItem] = field(default_factory=list)
    catalog: Dict[str, Any] = field(default_factory=dict)
    customer_name: str = ""
    order_stage: str = "greeting"  # greeting, browsing, cart_review, ordering, completed
    
    def get_cart_total(self) -> float:
        """Calculate total price of all items in cart."""
        return sum(item.total_price() for item in self.cart)
    
    def get_cart_item_count(self) -> int:
        """Get total number of items in cart."""
        return sum(item.quantity for item in self.cart)
    
    def find_cart_item(self, item_id: str) -> Optional[CartItem]:
        """Find a cart item by its ID."""
        for item in self.cart:
            if item.item_id == item_id:
                return item
        return None
    
    def remove_cart_item(self, item_id: str) -> bool:
        """Remove an item from the cart."""
        for i, item in enumerate(self.cart):
            if item.item_id == item_id:
                self.cart.pop(i)
                return True
        return False


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


class FoodOrderingAgent(BaseAgent):
    """Food & grocery ordering agent for FreshMart Express."""
    
    def __init__(self, chat_ctx: Optional[ChatContext] = None) -> None:
        instructions = """You are Aurora, a friendly and helpful shopping assistant for FreshMart Express, a food and grocery delivery service.

        Your role is to:
        1. Greet customers warmly and explain what you can help them with
        2. Help customers find and add items to their cart
        3. Handle intelligent requests like "ingredients for X" by adding multiple related items
        4. Manage their shopping cart (add, remove, update quantities, show contents)
        5. Process their final order when they're ready to checkout

        KEY CAPABILITIES:
        - Browse and search our full catalog of groceries, snacks, prepared foods, and beverages
        - Add specific items with quantities to their cart
        - Handle recipe-based requests intelligently (e.g., "I need ingredients for spaghetti dinner" or "ingredients for masala dosa")
        - Show cart contents with prices and totals
        - Remove or modify items in the cart
        - Complete orders and save them
        - Use rupees (₹) for prices and realistic local items

        INTELLIGENT FEATURES:
        - When customers ask for "ingredients for [dish]", ALWAYS use the find_recipe_ingredients function to get the raw ingredients needed to make that dish
        - For example: "ingredients for masala dosa" should give rice, dal, potatoes, oil - NOT the ready-made dosa
        - Always confirm what you're adding to the cart
        - Ask for clarification on quantities, sizes, or brands when needed
        - Keep track of cart total and item count
        - Suggest related or complementary items when appropriate

        CONVERSATION FLOW:
        1. Warm greeting and explain your services
        2. Help customer browse and add items to cart
        3. Handle any cart modifications they request
        4. When they're ready, review their final cart and place the order
        5. Confirm order placement with order details

        TONE: Friendly, helpful, and efficient. Make shopping feel easy and enjoyable!

        IMPORTANT: 
        - Always use the function tools to interact with the catalog and cart. Don't make up prices or items.
        - When someone asks for "ingredients for X", they want to cook it themselves, so give them the raw ingredients, not prepared food."""

        super().__init__(instructions=instructions, chat_ctx=chat_ctx)

    @function_tool
    async def browse_catalog(self, context: RunContext[OrderData], category: str = "") -> str:
        """Browse the food catalog by category or show all categories.
        
        Args:
            category: Optional category to browse (groceries, snacks, prepared_food, beverages)
        """
        catalog = context.userdata.catalog
        
        if not category:
            # Show all categories
            categories = list(catalog.get("catalog", {}).get("categories", {}).keys())
            return f"Welcome to FreshMart Express! We have these categories available: {', '.join(categories)}. What would you like to browse, or what specific items are you looking for?"
        
        category_lower = category.lower().replace(" ", "_")
        items = catalog.get("catalog", {}).get("categories", {}).get(category_lower, [])
        
        if not items:
            available_cats = list(catalog.get("catalog", {}).get("categories", {}).keys())
            return f"I don't see that category. We have: {', '.join(available_cats)}. What would you like to browse?"
        
        # Format items nicely
        item_list = []
        for item in items[:8]:  # Show first 8 items to avoid too long response
            item_list.append(f"• {item['name']} ({item['brand']}) - ₹{item['price']:.2f} - {item['size']}")
        
        result = f"Here are some {category} items:\n" + "\n".join(item_list)
        if len(items) > 8:
            result += f"\n... and {len(items) - 8} more items in this category."
        
        result += "\n\nWould you like me to add any of these to your cart, or search for something specific?"
        return result

    @function_tool
    async def search_items(self, context: RunContext[OrderData], search_term: str) -> str:
        """Search for items in the catalog by name or tags.
        
        Args:
            search_term: What to search for (item name, brand, or tag)
        """
        catalog = context.userdata.catalog
        search_lower = search_term.lower()
        found_items = []
        
        # Search through all categories
        for category_name, items in catalog.get("catalog", {}).get("categories", {}).items():
            for item in items:
                # Search in name, brand, and tags
                searchable_text = f"{item['name']} {item.get('brand', '')} {' '.join(item.get('tags', []))}"
                if search_lower in searchable_text.lower():
                    found_items.append(item)
        
        if not found_items:
            return f"I couldn't find any items matching '{search_term}'. Try browsing our categories or searching for something else like 'bread', 'pasta', 'milk', etc."
        
        # Format results
        item_list = []
        for item in found_items[:6]:  # Show first 6 matches
            item_list.append(f"• {item['name']} ({item['brand']}) - ₹{item['price']:.2f} - {item['size']} [Category: {item['category']}]")
        
        result = f"Found {len(found_items)} item(s) matching '{search_term}':\n" + "\n".join(item_list)
        if len(found_items) > 6:
            result += f"\n... and {len(found_items) - 6} more matches."
        
        result += "\n\nWould you like me to add any of these to your cart?"
        return result

    @function_tool
    async def add_to_cart(self, context: RunContext[OrderData], item_name: str, quantity: int = 1, notes: str = "") -> str:
        """Add an item to the shopping cart.
        
        Args:
            item_name: Name of the item to add
            quantity: How many to add (default 1)
            notes: Optional notes (size preference, brand, etc.)
        """
        catalog = context.userdata.catalog
        item_lower = item_name.lower()
        found_item = None
        
        # Search for the item in catalog
        for category_name, items in catalog.get("catalog", {}).get("categories", {}).items():
            for item in items:
                if item_lower in item['name'].lower() or item_lower in item.get('brand', '').lower():
                    found_item = item
                    break
            if found_item:
                break
        
        if not found_item:
            return f"I couldn't find '{item_name}' in our catalog. Try searching for it first or browse our categories."
        
        # Check if item already in cart
        existing_item = context.userdata.find_cart_item(found_item['id'])
        
        if existing_item:
            # Update quantity of existing item
            existing_item.quantity += quantity
            existing_item.notes = notes if notes else existing_item.notes
            cart_item = existing_item
        else:
            # Add new item to cart
            cart_item = CartItem(
                item_id=found_item['id'],
                name=found_item['name'],
                price=found_item['price'],
                quantity=quantity,
                category=found_item['category'],
                brand=found_item.get('brand', ''),
                size=found_item.get('size', ''),
                notes=notes
            )
            context.userdata.cart.append(cart_item)
        
        total_cost = cart_item.total_price()
        cart_total = context.userdata.get_cart_total()
        cart_count = context.userdata.get_cart_item_count()
        
        return f"Added {quantity} x {cart_item.name} to your cart! ₹{total_cost:.2f}\nYour cart now has {cart_count} items totaling ₹{cart_total:.2f}. Anything else you need?"

    @function_tool
    async def find_recipe_ingredients(self, context: RunContext[OrderData], dish_name: str) -> str:
        """Find and add ingredients for a specific dish/recipe.
        
        Args:
            dish_name: Name of the dish (e.g., "masala dosa", "spaghetti", "pasta")
        """
        catalog = context.userdata.catalog
        recipes = catalog.get("recipes", {})
        dish_lower = dish_name.lower().replace(" ", "_")
        
        # Find matching recipe
        recipe = None
        recipe_key = None
        
        # Try exact match first
        if dish_lower in recipes:
            recipe = recipes[dish_lower]
            recipe_key = dish_lower
        else:
            # Try partial matching - look for dish name in recipe keys
            for key, recipe_data in recipes.items():
                # Check if any word from dish_name matches recipe key or recipe name
                dish_words = dish_lower.split("_")
                key_words = key.split("_")
                recipe_name_words = recipe_data['name'].lower().replace(" ", "_").split("_")
                
                # Check for word overlap
                if (any(word in key_words for word in dish_words) or 
                    any(word in recipe_name_words for word in dish_words) or
                    dish_lower in key or dish_lower in recipe_data['name'].lower()):
                    recipe = recipe_data
                    recipe_key = key
                    break
        
        if not recipe:
            # Suggest available recipes
            available_recipes = [recipe_data['name'] for recipe_data in recipes.values()]
            return f"I don't have a recipe for '{dish_name}'. I can help with these dishes: {', '.join(available_recipes)}. Or you can add specific ingredients manually!"
        
        # Add all recipe ingredients to cart
        added_items = []
        total_added_cost = 0.0
        
        for ingredient_id in recipe['ingredients']:
            # Find the ingredient in catalog
            ingredient_item = None
            for category_items in catalog.get("catalog", {}).get("categories", {}).values():
                for item in category_items:
                    if item['id'] == ingredient_id:
                        ingredient_item = item
                        break
                if ingredient_item:
                    break
            
            if ingredient_item:
                # Check if already in cart
                existing_item = context.userdata.find_cart_item(ingredient_id)
                
                if existing_item:
                    existing_item.quantity += 1
                    cart_item = existing_item
                else:
                    cart_item = CartItem(
                        item_id=ingredient_item['id'],
                        name=ingredient_item['name'],
                        price=ingredient_item['price'],
                        quantity=1,
                        category=ingredient_item['category'],
                        brand=ingredient_item.get('brand', ''),
                        size=ingredient_item.get('size', ''),
                        notes=f"For {recipe['name']}"
                    )
                    context.userdata.cart.append(cart_item)
                
                added_items.append(f"• {cart_item.name} - ₹{cart_item.price:.2f}")
                total_added_cost += cart_item.price
        
        cart_total = context.userdata.get_cart_total()
        cart_count = context.userdata.get_cart_item_count()
        
        result = f"Perfect! I've added ingredients for {recipe['name']} to your cart:\n"
        result += "\n".join(added_items)
        result += f"\n\nAdded ₹{total_added_cost:.2f} worth of ingredients."
        result += f"\nYour cart now has {cart_count} items totaling ₹{cart_total:.2f}. Need anything else?"
        
        return result

    @function_tool
    async def show_cart(self, context: RunContext[OrderData]) -> str:
        """Show current cart contents with totals."""
        cart = context.userdata.cart
        
        if not cart:
            return "Your cart is currently empty. Would you like to browse our categories or search for specific items?"
        
        cart_items = []
        total = 0.0
        
        for item in cart:
            item_total = item.total_price()
            total += item_total
            
            item_str = f"• {item.quantity} x {item.name}"
            if item.brand:
                item_str += f" ({item.brand})"
            if item.size:
                item_str += f" - {item.size}"
            item_str += f" = ₹{item_total:.2f}"
            if item.notes:
                item_str += f" [{item.notes}]"
            
            cart_items.append(item_str)
        
        result = f"🛒 Your Cart ({len(cart)} items):\n"
        result += "\n".join(cart_items)
        result += f"\n\nTotal: ₹{total:.2f}"
        result += "\n\nWould you like to add more items, modify quantities, or place your order?"
        
        return result

    @function_tool
    async def remove_from_cart(self, context: RunContext[OrderData], item_name: str) -> str:
        """Remove an item from the cart.
        
        Args:
            item_name: Name of the item to remove
        """
        item_lower = item_name.lower()
        removed_item = None
        
        for item in context.userdata.cart:
            if item_lower in item.name.lower():
                removed_item = item
                context.userdata.remove_cart_item(item.item_id)
                break
        
        if not removed_item:
            return f"I couldn't find '{item_name}' in your cart. Here's what you currently have: {', '.join([item.name for item in context.userdata.cart])}"
        
        cart_total = context.userdata.get_cart_total()
        cart_count = context.userdata.get_cart_item_count()
        
        return f"Removed {removed_item.name} from your cart. Your cart now has {cart_count} items totaling ₹{cart_total:.2f}."

    @function_tool
    async def update_cart_quantity(self, context: RunContext[OrderData], item_name: str, new_quantity: int) -> str:
        """Update the quantity of an item in the cart.
        
        Args:
            item_name: Name of the item to update
            new_quantity: New quantity (use 0 to remove item)
        """
        if new_quantity < 0:
            return "Quantity must be 0 or positive. Use 0 to remove the item."
        
        item_lower = item_name.lower()
        target_item = None
        
        for item in context.userdata.cart:
            if item_lower in item.name.lower():
                target_item = item
                break
        
        if not target_item:
            return f"I couldn't find '{item_name}' in your cart. Here's what you currently have: {', '.join([item.name for item in context.userdata.cart])}"
        
        if new_quantity == 0:
            context.userdata.remove_cart_item(target_item.item_id)
            cart_total = context.userdata.get_cart_total()
            cart_count = context.userdata.get_cart_item_count()
            return f"Removed {target_item.name} from your cart. Your cart now has {cart_count} items totaling ₹{cart_total:.2f}."
        else:
            old_quantity = target_item.quantity
            target_item.quantity = new_quantity
            
            cart_total = context.userdata.get_cart_total()
            cart_count = context.userdata.get_cart_item_count()
            
            return f"Updated {target_item.name} quantity from {old_quantity} to {new_quantity}. Your cart now has {cart_count} items totaling ₹{cart_total:.2f}."

    @function_tool
    async def place_order(self, context: RunContext[OrderData], customer_name: str = "") -> str:
        """Complete the order and save it to a JSON file.
        
        Args:
            customer_name: Optional customer name for the order
        """
        cart = context.userdata.cart
        
        if not cart:
            return "Your cart is empty! Add some items first before placing an order."
        
        if customer_name:
            context.userdata.customer_name = customer_name
        
        # Create order object
        order_id = str(uuid4())[:8]
        order = {
            "order_id": order_id,
            "store_name": "FreshMart Express",
            "customer_name": context.userdata.customer_name or "Guest Customer",
            "order_timestamp": datetime.now().isoformat(),
            "items": [item.to_dict() for item in cart],
            "item_count": context.userdata.get_cart_item_count(),
            "total_amount": context.userdata.get_cart_total(),
            "status": "confirmed",
            "estimated_delivery": "30-45 minutes"
        }
        
        try:
            # Save order to file
            order_file = save_order_to_file(order)
            
            # Clear the cart
            context.userdata.cart.clear()
            context.userdata.order_stage = "completed"
            
            result = f"🎉 Order placed successfully!\n\n"
            result += f"Order ID: {order_id}\n"
            result += f"Customer: {order['customer_name']}\n"
            result += f"Items: {order['item_count']} items\n"
            result += f"Total: ₹{order['total_amount']:.2f}\n"
            result += f"Estimated Delivery: {order['estimated_delivery']}\n\n"
            result += f"Your order has been saved and will be prepared shortly. "
            result += f"Thank you for choosing FreshMart Express! 🛒✨"
            
            return result
            
        except Exception as e:
            logger.error(f"Error placing order: {e}")
            return "Sorry, there was an issue placing your order. Please try again or contact support."

    @function_tool
    async def get_help(self, context: RunContext[OrderData]) -> str:
        """Show help information about what the assistant can do."""
        help_text = """🛒 Welcome to FreshMart Express! Here's what I can help you with:

        📋 BROWSING & SEARCHING:
        • "Show me groceries" - Browse by category
        • "Search for bread" - Find specific items
        • "What do you have?" - See all categories

        🥪 SMART ORDERING:
        • "I need ingredients for spaghetti" - Get recipe ingredients
        • "Add pasta to my cart" - Add specific items
        • "Add 2 bottles of milk" - Specify quantities

        🛒 CART MANAGEMENT:
        • "What's in my cart?" - See cart contents
        • "Remove bread from cart" - Remove items
        • "Change milk quantity to 3" - Update amounts

        ✅ CHECKOUT:
        • "I'm ready to order" - Place your order
        • "That's all" - Complete your shopping

        Just tell me what you need and I'll help you find it! 😊"""
        
        return help_text


def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


async def entrypoint(ctx: JobContext):
    # Load the food catalog
    catalog = load_food_catalog()
    logger.info(f"Loaded food catalog with {len(catalog.get('catalog', {}).get('categories', {}))} categories")
    
    # Logging setup
    ctx.log_context_fields = {
        "room": ctx.room.name,
    }

    # Initialize Order data
    userdata = OrderData(catalog=catalog)
    
    # Create Food Ordering agent
    food_agent = FoodOrderingAgent()
    
    # Create the session with the Food Ordering agent
    session = AgentSession[OrderData](
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

    # Start the session with the food agent
    await session.start(
        agent=food_agent,
        room=ctx.room,
        room_input_options=RoomInputOptions(
            noise_cancellation=noise_cancellation.BVC(),
        ),
    )

    # Join the room and connect to the user
    await ctx.connect()


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))
