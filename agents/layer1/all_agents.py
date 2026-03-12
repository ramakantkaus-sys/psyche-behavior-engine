"""
PSYCHE OS — Layer 1 Agents Re-export
Re-exports all 26 agents, build_all_agents, and DOMAIN_GROUPS
so that `from agents.layer1.all_agents import build_all_agents, DOMAIN_GROUPS` works.
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from psyche_layer1_agents import *
from psyche_layer1_agents import build_all_agents, DOMAIN_GROUPS
