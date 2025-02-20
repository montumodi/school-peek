#!/bin/sh
gunicorn -w 1 -b 0.0.0.0:${PORT:-5000} src.api.server:app
