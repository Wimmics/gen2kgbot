"""
This module implements the Langgraph contitional edges, aka. routers,
that are common to multiple scenarios
"""

import ast
from typing import List
import os
from app.core.utils.graph_state import OverallState
import app.core.utils.config_manager as config

from app.core.utils.construct_util import (
    generate_class_context_filename,
)
from langgraph.constants import Send


logger = config.setup_logger(__package__, __file__)


def get_class_context_router(
    state: OverallState,
) -> List[Send]:
    """
    Looks in the cache folder whether the context of the selected similar classes are
    already present. If not, route to the node that will fetch the context from the knowledge graph.

    This function must be invoked after classes similar to the user question were set in OverallState.selected_classes.

    Args:
        state (dict): current state of the conversation

    Returns:
        List[Send]: node to be executed next with class context file path.
            Next node should be one of "get_context_class_from_cache" or "get_context_class_from_kg".
            If "get_context_class_from_cache", the additional arg is the file path.
            If "get_context_class_from_kg", the additional arg is a tuple (uri, label, description).
    """
    next_nodes = []

    logger.info("Looking for class contexts (in ttl) from the cache folder...")

    for item in state["selected_classes"]:
        cls = ast.literal_eval(item)
        cls_path = generate_class_context_filename(cls[0])

        if os.path.exists(cls_path):
            logger.debug(f"Class context found in cache: {cls_path}.")
            next_nodes.append(Send("get_context_class_from_cache", cls_path))
        else:
            logger.debug(f"Class context not found in cache for class {cls}.")
            next_nodes.append(Send("get_context_class_from_kg", cls))

    return next_nodes
