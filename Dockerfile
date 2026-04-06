ARG BUILD_FROM
FROM ${BUILD_FROM}

ENV LANG=C.UTF-8

# Install system + build dependencies
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-dev \
    python3-numpy \
    gcc \
    g++ \
    libffi-dev \
    libsndfile1 \
    libsndfile1-dev \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Install PyTorch CPU-only first (avoids pulling ~2GB of CUDA packages)
RUN pip3 install --no-cache-dir --break-system-packages \
    torch --index-url https://download.pytorch.org/whl/cpu

# Install remaining Python dependencies
COPY requirements.txt /tmp/
RUN pip3 install --no-cache-dir --break-system-packages -r /tmp/requirements.txt \
    && rm /tmp/requirements.txt

# Pre-download the resemblyzer model
RUN python3 -c "from resemblyzer import VoiceEncoder; VoiceEncoder('cpu')"

# Copy application code
COPY speaker_id/ /usr/src/speaker_id/
COPY rootfs/ /

# Create persistent data directories
RUN mkdir -p /data/profiles /data/enrollment_audio

# Fix Windows CRLF -> Unix LF and set executable permissions
RUN find /etc/s6-overlay -type f -exec sed -i 's/\r$//' {} + \
    && chmod +x /etc/s6-overlay/s6-rc.d/speaker-id/run

# Set workdir so python -m speaker_id works
ENV PYTHONPATH="/usr/src"
WORKDIR /usr/src

# Ensure s6-overlay runs as PID 1
ENTRYPOINT ["/init"]

EXPOSE 10310 8756

LABEL \
    io.hass.name="Wyoming Speaker ID" \
    io.hass.description="Speaker recognition for HA voice pipeline" \
    io.hass.type="addon" \
    io.hass.version="0.2.0"
