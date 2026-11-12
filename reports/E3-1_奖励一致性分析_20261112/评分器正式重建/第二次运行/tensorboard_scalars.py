#!/usr/bin/env python3
"""Read scalar Event protobufs and verify TFRecord framing without a live directory watcher."""
import json
import struct
from pathlib import Path

from tensorboard.compat.proto.event_pb2 import Event

CRC_TABLE = []
for initial in range(256):
    value = initial
    for _ in range(8):
        value = (value >> 1) ^ (0x82F63B78 if value & 1 else 0)
    CRC_TABLE.append(value)


def masked_crc32c(data):
    value = 0xFFFFFFFF
    for byte in data:
        value = CRC_TABLE[(value ^ byte) & 0xFF] ^ (value >> 8)
    value ^= 0xFFFFFFFF
    return (((value >> 15) | (value << 17)) + 0xA282EAD8) & 0xFFFFFFFF


def read_scalars(directory, wanted_tags=None, allow_partial_tail=False):
    scalars = {tag: [] for tag in wanted_tags} if wanted_tags is not None else {}
    record_count = 0
    for path in sorted(Path(directory).glob('events.out.tfevents.*')):
        data = path.read_bytes()
        offset = 0
        while offset < len(data):
            if len(data)-offset < 12:
                assert allow_partial_tail, (path.name, offset, 'incomplete header')
                break
            header = data[offset:offset+8]
            length = struct.unpack('<Q', header)[0]
            header_crc = struct.unpack('<I', data[offset+8:offset+12])[0]
            assert masked_crc32c(header) == header_crc, (path.name, offset, 'header CRC')
            if offset+16+length > len(data):
                assert allow_partial_tail, (path.name, offset, 'incomplete payload')
                break
            payload = data[offset+12:offset+12+length]
            payload_crc = struct.unpack('<I', data[offset+12+length:offset+16+length])[0]
            assert masked_crc32c(payload) == payload_crc, (path.name, offset, 'payload CRC')
            event = Event.FromString(payload)
            for item in event.summary.value:
                if wanted_tags is not None and item.tag not in wanted_tags:
                    continue
                if not item.HasField('simple_value'):
                    assert wanted_tags is None, (item.tag, 'expected native simple_value scalar')
                    continue
                scalars.setdefault(item.tag, []).append({'step': event.step, 'value': item.simple_value})
            record_count += 1
            offset += 16+length
    return scalars, record_count


if __name__ == '__main__':
    directory = '/private/AGSE_E3_1_rebuild_20261112/scorer_training_20261112/attempt_02/run_01/logs/plain_prior_scorer/run_01'
    scalars, count = read_scalars(directory, allow_partial_tail=True)
    print(json.dumps({'crc_verified_records': count,
        'recall_scalar_counts': {tag: len(values) for tag, values in scalars.items() if tag.startswith('recalls/')}}))
