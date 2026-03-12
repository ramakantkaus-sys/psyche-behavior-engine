"""
PSYCHE OS — Base Agent Re-export
Re-exports MicroAgent from the root psyche_base_agent module
so that `from agents.base_agent import MicroAgent` works.
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from psyche_base_agent import *
from psyche_base_agent import MicroAgent
