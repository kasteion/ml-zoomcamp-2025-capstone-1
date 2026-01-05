FROM python:3.13.1-slim

# RUN pip install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

ENV PATH="/app/.venv/bin:$PATH"

COPY ".python-version" "pyproject.toml" "uv.lock" ./
RUN uv sync --no-dev

COPY sentiment_model/ ./sentiment_model/
COPY "predict.py" ./

EXPOSE 9696

ENTRYPOINT ["uvicorn", "predict:app", "--host", "0.0.0.0", "--port", "9696" ]
