FROM python:3.12-slim-bookworm

RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates

ADD https://astral.sh/uv/install.sh /uv-installer.sh
RUN sh /uv-installer.sh && rm /uv-installer.sh
ENV PATH="/root/.local/bin/:$PATH"

WORKDIR /code

COPY pyproject.toml uv.lock /code/
RUN uv sync --frozen --no-install-project

COPY ./app /code/app

CMD ["uv", "run", "--no-sync", "fastapi", "run", "app/main.py", "--port", "80"]