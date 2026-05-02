"""
Pytest configuration. Adds `agents/` to sys.path so tests can import the
agent base classes without a full install. We do NOT depend on the generated
gRPC stubs being present — agent imports tolerate them being missing.
"""
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "agents"))
