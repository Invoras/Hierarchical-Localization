#!/usr/bin/env python3
"""
Simple LiveKit frame capture script for drone mapping
Connects to the drone-stream room and saves frames every 0.5 seconds
"""

import asyncio
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
import requests
import cv2
import numpy as np
from livekit import rtc

# Configuration
ROOM_NAME = "drone-stream"
TOKEN_SERVER_URL = "https://n189ebyfq6.execute-api.eu-central-1.amazonaws.com/default/token"
FRAME_INTERVAL = 0.1  # seconds
WARMUP_DELAY = 10  # seconds - wait for SFU bandwidth estimation to ramp up to full quality

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class FrameCapture:
    def __init__(self, output_folder: str):
        self.output_folder = Path(output_folder)
        self.output_folder.mkdir(parents=True, exist_ok=True)
        self.frame_count = 0
        self.last_save_time = 0
        self.running = True

        logger.info(f"Output folder: {self.output_folder}")

    async def connect_to_room(self):
        """Fetch token and connect to LiveKit room"""
        try:
            # Fetch token from AWS Lambda
            logger.info(f"Fetching token from {TOKEN_SERVER_URL}...")
            response = requests.post(
                TOKEN_SERVER_URL,
                json={
                    "roomName": ROOM_NAME,
                    "participantName": "frame-capture-client",
                    "isPublisher": False
                },
                timeout=10
            )
            response.raise_for_status()

            data = response.json()
            url = data["url"]
            token = data["token"]

            logger.info(f"Connecting to room: {ROOM_NAME}")

            # Connect to LiveKit room
            room = rtc.Room()

            # Set up event handlers
            @room.on("track_subscribed")
            def on_track_subscribed(track: rtc.Track, publication: rtc.TrackPublication, participant: rtc.RemoteParticipant):
                logger.info(f"Track subscribed: {track.kind} from {participant.identity}")

                if track.kind == rtc.TrackKind.KIND_VIDEO:
                    # Log detailed track information
                    logger.info(f"📺 Video Track Details:")
                    logger.info(f"   Publication resolution: {publication.width}x{publication.height}")
                    logger.info(f"   Track SID: {publication.sid}")
                    logger.info(f"   Track Name: {publication.name}")
                    logger.info(f"   Simulcasted: {publication.simulcasted}")
                    logger.info(f"   MIME Type: {publication.mime_type}")

                    asyncio.create_task(self.process_video_track(track, publication))

            @room.on("participant_connected")
            def on_participant_connected(participant: rtc.RemoteParticipant):
                logger.info(f"Participant connected: {participant.identity}")

            @room.on("participant_disconnected")
            def on_participant_disconnected(participant: rtc.RemoteParticipant):
                logger.info(f"Participant disconnected: {participant.identity}")

            @room.on("data_received")
            def on_data_received(data: rtc.DataPacket):
                """Receive telemetry data packets"""
                try:
                    telemetry = json.loads(data.data.decode('utf-8'))

                    positioning = telemetry.get('positioning_source', 'UNKNOWN')
                    gps_sats = telemetry.get('gps_satellite_count', 0)
                    gps_signal = telemetry.get('gps_signal_level', 0)

                    # VPS position (meters from takeoff when using VPS)
                    pos_x = telemetry.get('position_x_m', 0.0)
                    pos_y = telemetry.get('position_y_m', 0.0)
                    pos_z = telemetry.get('position_z_m', 0.0)

                    # Warn if not using VPS in warehouse
                    if positioning != 'VPS':
                        logger.warning(f"⚠️  NOT using VPS! Current: {positioning}")

                    logger.info(
                        f"📊 Telemetry | "
                        f"Frame: {telemetry['frame_number']:06d} | "
                        f"Src: {positioning} "
                        f"(GPS: {gps_sats}sat/sig{gps_signal}) | "
                        f"VPS Pos: ({pos_x:.2f}, {pos_y:.2f}, {pos_z:.2f})m | "
                        f"Vel: ({telemetry['velocity_x_ms']:.2f}, "
                        f"{telemetry['velocity_y_ms']:.2f}, "
                        f"{telemetry['velocity_z_ms']:.2f}) m/s | "
                        f"Hdg: {telemetry['heading_deg']:.1f}°"
                    )
                except Exception as e:
                    logger.error(f"Failed to parse telemetry: {e}")

            # Connect with room options
            # Note: adaptive_stream is disabled by default in Python SDK when not using UI elements
            room_options = rtc.RoomOptions(
                auto_subscribe=True,  # Automatically subscribe to tracks
                dynacast=False,  # Disable dynamic quality - we want full resolution always
            )
            await room.connect(url, token, options=room_options)
            logger.info("✅ Connected to LiveKit room!")
            logger.info("   (Dynacast disabled - requesting full quality from start)")

            return room

        except requests.RequestException as e:
            logger.error(f"Failed to fetch token: {e}")
            raise
        except Exception as e:
            logger.error(f"Failed to connect to room: {e}")
            raise

    async def process_video_track(self, track: rtc.VideoTrack, publication: rtc.TrackPublication):
        """Process incoming video frames"""
        logger.info("🎥 Video track received, starting warm-up period...")

        video_stream = rtc.VideoStream(track)
        first_frame = True
        warmup_complete = False
        warmup_start_time = None
        last_countdown_log = -1

        async for frame_event in video_stream:
            if not self.running:
                break

            # Log first frame details
            if first_frame:
                frame = frame_event.frame
                logger.info(f"🎬 First frame received:")
                logger.info(f"   Frame resolution: {frame.width}x{frame.height}")
                logger.info(f"   Publication resolution: {publication.width}x{publication.height}")
                logger.info(f"")
                logger.info(f"⏳ Warming up connection for {WARMUP_DELAY}s to allow SFU quality ramp-up...")
                warmup_start_time = asyncio.get_event_loop().time()
                first_frame = False

            # Warm-up period: wait for SFU to ramp up quality
            if not warmup_complete:
                current_time = asyncio.get_event_loop().time()
                elapsed = current_time - warmup_start_time

                # Log countdown every 5 seconds (avoid duplicate logs)
                elapsed_rounded = int(elapsed)
                if elapsed_rounded % 5 == 0 and elapsed_rounded > 0 and elapsed_rounded != last_countdown_log:
                    remaining = WARMUP_DELAY - elapsed
                    if remaining > 0:
                        logger.info(f"   Warm-up: {int(remaining)}s remaining...")
                        last_countdown_log = elapsed_rounded

                # Check if warm-up is complete
                if elapsed >= WARMUP_DELAY:
                    frame = frame_event.frame
                    logger.info(f"")
                    logger.info(f"✅ Warm-up complete! Final resolution check:")
                    logger.info(f"   Frame resolution: {frame.width}x{frame.height}")

                    if frame.width != 1920 or frame.height != 1080:
                        logger.warning(f"⚠️  WARNING: Still receiving {frame.width}x{frame.height} instead of 1920x1080")
                        logger.warning(f"   SFU may need more time or connection bandwidth is limited")
                    else:
                        logger.info(f"✅ SUCCESS: Receiving full 1080p resolution!")

                    logger.info(f"")
                    logger.info(f"📹 Starting frame capture every {FRAME_INTERVAL}s...")
                    warmup_complete = True
                    self.last_save_time = asyncio.get_event_loop().time()
                continue

            # Normal frame capture after warm-up
            current_time = asyncio.get_event_loop().time()

            # Save frame every FRAME_INTERVAL seconds
            if current_time - self.last_save_time >= FRAME_INTERVAL:
                await self.save_frame(frame_event.frame)
                self.last_save_time = current_time

    async def save_frame(self, frame: rtc.VideoFrame):
        """Convert and save video frame as image"""
        try:
            # Convert to numpy array (I420 -> BGR)
            buffer = frame.convert(rtc.VideoBufferType.RGBA)

            # Get frame data as numpy array
            data = np.frombuffer(buffer.data, dtype=np.uint8)
            img = data.reshape((buffer.height, buffer.width, 4))

            # Convert RGBA to BGR for OpenCV
            img_bgr = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)

            # Generate filename with frame number and timestamp
            timestamp = datetime.now().strftime("%H-%M-%S-%f")[:-3]  # milliseconds
            filename = f"frame_{self.frame_count:06d}_{timestamp}.jpg"
            filepath = self.output_folder / filename

            # Save image
            cv2.imwrite(str(filepath), img_bgr)

            self.frame_count += 1
            if self.frame_count % 10 == 0:
            	logger.info(f"💾 Saved frame {self.frame_count}: {filename}")

        except Exception as e:
            logger.error(f"Failed to save frame: {e}")

    async def run(self):
        """Main run loop"""
        room = None
        try:
            room = await self.connect_to_room()

            logger.info(f"📹 Capturing frames every {FRAME_INTERVAL}s. Press Ctrl+C to stop.")

            # Keep running until interrupted
            while self.running:
                await asyncio.sleep(1)

        except KeyboardInterrupt:
            logger.info("\n🛑 Stopping capture...")
            self.running = False
        except Exception as e:
            logger.error(f"Error: {e}")
            raise
        finally:
            if room:
                await room.disconnect()
            logger.info(f"✅ Capture complete! Saved {self.frame_count} frames to {self.output_folder}")


async def main():
    """Entry point"""
    # Create output folder with timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_folder = f"captured_frames_{timestamp}"

    logger.info("=" * 60)
    logger.info("🚁 Drone Frame Capture")
    logger.info("=" * 60)

    # Create and run capture
    capture = FrameCapture(output_folder)
    await capture.run()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        sys.exit(0)
