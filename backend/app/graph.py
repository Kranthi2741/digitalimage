from langgraph.graph import StateGraph, END
import json
from pathlib import Path

from app.state import CampaignState
from app.agents.planner import planner_node
from app.agents.strategist import strategist_node
from app.agents.creative_director import creative_director_node
from app.agents.image_generator import image_generator_node
from app.agents.quality_agent import quality_agent_node, route_after_quality
from app.agents.content_agent import content_agent_node
from app.agents.publisher import publisher_node
from app.agents.video_producer import video_producer_node
from app.agents.asset_library import asset_library_node
from app.agents.compositor import compositor_node


def _saved_brand_profile() -> dict:
    """Load the admin-selected brand kit for every campaign entry point."""
    path = Path(__file__).resolve().parents[1] / "app_data" / "settings.json"
    try:
        settings = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    except (OSError, ValueError):
        settings = {}
    return {
        "name": settings.get("brand_name", ""),
        "tagline": settings.get("brand_tagline", ""),
        "primary_color": settings.get("brand_primary_color", "#283E53"),
        "accent_color": settings.get("brand_accent_color", "#ED3237"),
        "cta_color": settings.get("brand_cta_color", "#C5222B"),
        "supporting_color": settings.get("brand_supporting_color", "#137C80"),
        "surface_color": settings.get("brand_surface_color", "#F3F6F8"),
        "font": settings.get("brand_font", "Exo"),
        "heading_font": settings.get("brand_heading_font", settings.get("brand_font", "Exo")),
        "body_font": settings.get("brand_body_font", "Manrope"),
        "data_font": settings.get("brand_data_font", "Source Sans 3"),
        "visual_style": settings.get("brand_visual_style", "Clinical editorial"),
        "healthcare_tone": settings.get("brand_healthcare_tone", "Clear, empathetic and trustworthy"),
    }


def build_graph():
    graph = StateGraph(CampaignState)

    graph.add_node("planner", planner_node)
    graph.add_node("strategist", strategist_node)
    graph.add_node("creative_director", creative_director_node)
    graph.add_node("image_generator", image_generator_node)
    graph.add_node("quality_agent", quality_agent_node)
    graph.add_node("content_agent", content_agent_node)
    graph.add_node("publisher", publisher_node)
    graph.add_node("video_producer", video_producer_node)
    graph.add_node("asset_library", asset_library_node)
    graph.add_node("compositor", compositor_node)

    graph.set_entry_point("planner")

    graph.add_edge("planner", "strategist")
    graph.add_edge("strategist", "creative_director")
    graph.add_edge("creative_director", "image_generator")
    graph.add_edge("image_generator", "quality_agent")

    # this is the core loop: pass/fail/give-up routing
    graph.add_conditional_edges(
        "quality_agent",
        route_after_quality,
        {
            "publisher": "content_agent",   # approved -> write caption first
            "creative_director": "creative_director",  # failed -> retry with feedback
            "end_failed": END,              # ran out of retries -> stop
        },
    )

    graph.add_edge("content_agent", "asset_library")
    graph.add_edge("asset_library", "compositor")
    graph.add_conditional_edges(
        "compositor",
        route_after_content,
        {
            "video_producer": "video_producer",
            "publisher": "publisher",
        },
    )
    graph.add_edge("video_producer", "publisher")
    graph.add_edge("publisher", END)

    return graph.compile()


def route_after_content(state: CampaignState) -> str:
    """Route to video_producer only when the prompt explicitly requests a video/reel/animation."""
    prompt = (state.get("user_prompt") or "").lower()
    video_keywords = ("video", "reel", "animation", "motion graphic", "mp4", "short film")
    return "video_producer" if any(k in prompt for k in video_keywords) else "publisher"


def run_campaign(user_prompt: str, business_profile: dict = None, max_retries: int = 3,
                 scheduled_date: str = None, auto_publish: bool = True, source_image_path: str = None, language: str = "English"):
    app = build_graph()
    profile = _saved_brand_profile()
    profile.update(business_profile or {})
    initial_state = {
        "user_prompt": user_prompt,
        "business_profile": profile,
        "retry_count": 0,
        "max_retries": max_retries,
        "scheduled_date": scheduled_date,
        "auto_publish": auto_publish,
        "source_image_path": source_image_path,
        "language": language,
    }
    final_state = app.invoke(initial_state)
    return final_state


def stream_campaign(user_prompt: str, business_profile: dict = None, max_retries: int = 3,
                    scheduled_date: str = None, auto_publish: bool = True, source_image_path: str = None, language: str = "English"):
    """Same as run_campaign but yields (node_name, state_after_node) after every
    step, so a UI can show live progress instead of waiting for the whole thing."""
    app = build_graph()
    profile = _saved_brand_profile()
    profile.update(business_profile or {})
    initial_state = {
        "user_prompt": user_prompt,
        "business_profile": profile,
        "retry_count": 0,
        "max_retries": max_retries,
        "scheduled_date": scheduled_date,
        "auto_publish": auto_publish,
        "source_image_path": source_image_path,
        "language": language,
    }
    for step in app.stream(initial_state):
        # step is like {"planner": {...partial state...}}
        for node_name, node_state in step.items():
            yield node_name, node_state
