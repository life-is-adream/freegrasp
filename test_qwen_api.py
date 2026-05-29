#!/usr/bin/env python3
"""Test script for Qwen3-VL-Flash API call via DashScope."""

import os
import sys
from openai import OpenAI


def test_api():
    api_key = os.environ.get("DASHSCOPE_API_KEY", "")
    if not api_key:
        print("ERROR: DASHSCOPE_API_KEY not set!")
        print('Run: export DASHSCOPE_API_KEY="sk-your-key-here"')
        sys.exit(1)

    client = OpenAI(
        api_key=api_key,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )

    # Test 1: Text-only call
    print("=== Test 1: Text-only call ===")
    try:
        completion = client.chat.completions.create(
            model="qwen3-vl-flash",
            messages=[{"role": "user", "content": "Hello, who are you?"}],
        )
        print("Response:", completion.choices[0].message.content)
        print("✅ Text call SUCCESS\n")
    except Exception as e:
        print(f"❌ Text call FAILED: {e}\n")
        return

    # Test 2: Vision call with image URL
    print("=== Test 2: Vision call with image ===")
    try:
        completion = client.chat.completions.create(
            model="qwen3-vl-flash",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": "https://help-static-aliyun-doc.aliyuncs.com/file-manage-files/zh-CN/20241022/emyrja/dog_and_girl.jpeg"
                            },
                        },
                        {"type": "text", "text": "What is shown in this image?"},
                    ],
                },
            ],
        )
        print("Response:", completion.choices[0].message.content)
        print("✅ Vision call SUCCESS")
    except Exception as e:
        print(f"❌ Vision call FAILED: {e}")


if __name__ == "__main__":
    test_api()
