"""WSGI entry point: ``gunicorn wsgi:application``."""

from ehri_cate.app import app as application
