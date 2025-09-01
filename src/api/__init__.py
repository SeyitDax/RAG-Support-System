"""
API Module

Flask-based REST API for customer support automation with confidence-based routing
and integration endpoints for n8n workflows.
"""

from .app import create_app

__all__ = ["create_app"]