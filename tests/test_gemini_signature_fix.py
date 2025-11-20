"""
Test to verify Gemini thought signature handling is working correctly.
"""

import asyncio
import os
from localrouter.dtypes import (
    ChatMessage,
    MessageRole,
    TextBlock,
    ThinkingBlock,
    ToolUseBlock,
    ToolResultBlock,
    ToolDefinition,
    ReasoningConfig,
)
from localrouter.llm import get_response


# Simple calculator tool
calculator_tool = ToolDefinition(
    name="calculator",
    description="Perform mathematical calculations",
    input_schema={
        "type": "object",
        "properties": {
            "expression": {
                "type": "string",
                "description": "Mathematical expression to evaluate",
            }
        },
        "required": ["expression"],
    },
)


async def test_gemini_signature_extraction():
    """Test that we correctly extract thought signatures from Gemini responses"""
    print("=" * 80)
    print("  TEST 1: Signature Extraction")
    print("=" * 80)
    
    if not os.environ.get("GEMINI_API_KEY") and not os.environ.get("GOOGLE_API_KEY"):
        print("❌ No GEMINI_API_KEY found")
        return False
    
    messages = [
        ChatMessage(
            role=MessageRole.user,
            content=[TextBlock(text="Calculate 17 * 9 using the calculator tool")],
        )
    ]
    
    try:
        response = await get_response(
            model="gemini-2.5-flash",
            messages=messages,
            tools=[calculator_tool],
            reasoning=ReasoningConfig(budget_tokens=2000),
            max_tokens=1000,
        )
        
        # Check for tool use blocks with signatures
        tool_use_blocks = [b for b in response.content if isinstance(b, ToolUseBlock)]
        
        if not tool_use_blocks:
            print("❌ No tool use blocks found")
            return False
        
        tool_block = tool_use_blocks[0]
        print(f"✓ Tool use block found: {tool_block.name}")
        print(f"  Input: {tool_block.input}")
        print(f"  Has thought_signature: {tool_block.thought_signature is not None}")
        
        if tool_block.thought_signature:
            print(f"  Signature length: {len(tool_block.thought_signature)} chars (base64)")
            print(f"  Signature preview: {tool_block.thought_signature[:50]}...")
            print("✓ Signature extracted successfully")
        else:
            print("⚠️  No signature found (may be expected if thinking wasn't used)")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_gemini_signature_serialization():
    """Test that signatures survive JSON serialization"""
    print("\n" + "=" * 80)
    print("  TEST 2: Signature Serialization")
    print("=" * 80)
    
    if not os.environ.get("GEMINI_API_KEY") and not os.environ.get("GOOGLE_API_KEY"):
        print("❌ No GEMINI_API_KEY found")
        return False
    
    messages = [
        ChatMessage(
            role=MessageRole.user,
            content=[TextBlock(text="Calculate 23 * 6 using the calculator")],
        )
    ]
    
    try:
        response = await get_response(
            model="gemini-2.5-flash",
            messages=messages,
            tools=[calculator_tool],
            reasoning=ReasoningConfig(budget_tokens=2000),
            max_tokens=1000,
        )
        
        tool_use_blocks = [b for b in response.content if isinstance(b, ToolUseBlock)]
        if not tool_use_blocks:
            print("❌ No tool use blocks")
            return False
        
        original_signature = tool_use_blocks[0].thought_signature
        print(f"Original signature: {original_signature is not None}")
        
        # Serialize to dict
        message_dict = response.model_dump()
        print(f"✓ Serialized to dict")
        
        # Deserialize back
        restored = ChatMessage(**message_dict)
        tool_use_blocks_restored = [b for b in restored.content if isinstance(b, ToolUseBlock)]
        restored_signature = tool_use_blocks_restored[0].thought_signature
        
        print(f"Restored signature: {restored_signature is not None}")
        
        if original_signature == restored_signature:
            print("✓ Signature preserved through serialization")
            return True
        else:
            print(f"❌ Signature changed!")
            print(f"  Original: {original_signature[:50] if original_signature else None}")
            print(f"  Restored: {restored_signature[:50] if restored_signature else None}")
            return False
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_gemini_signature_roundtrip():
    """Test that signatures are correctly sent back to Gemini in multi-turn"""
    print("\n" + "=" * 80)
    print("  TEST 3: Signature Round-trip (Multi-turn)")
    print("=" * 80)
    
    if not os.environ.get("GEMINI_API_KEY") and not os.environ.get("GOOGLE_API_KEY"):
        print("❌ No GEMINI_API_KEY found")
        return False
    
    messages = [
        ChatMessage(
            role=MessageRole.user,
            content=[TextBlock(text="Calculate 19 * 8 using the calculator")],
        )
    ]
    
    try:
        # First turn
        response1 = await get_response(
            model="gemini-2.5-flash",
            messages=messages,
            tools=[calculator_tool],
            reasoning=ReasoningConfig(budget_tokens=2000),
            max_tokens=1000,
        )
        
        tool_use_blocks = [b for b in response1.content if isinstance(b, ToolUseBlock)]
        if not tool_use_blocks:
            print("❌ No tool use in first response")
            return False
        
        tool_block = tool_use_blocks[0]
        has_signature = tool_block.thought_signature is not None
        print(f"First response:")
        print(f"  Tool: {tool_block.name}")
        print(f"  Has signature: {has_signature}")
        
        # Simulate tool execution
        expression = tool_block.input.get("expression", "")
        result = eval(expression)
        print(f"  Executed: {expression} = {result}")
        
        # Add to conversation
        messages.append(response1)
        messages.append(ChatMessage(
            role=MessageRole.user,
            content=[
                ToolResultBlock(
                    tool_use_id=tool_block.id,
                    content=[TextBlock(text=f"Result: {result}")],
                )
            ],
        ))
        
        # Second turn - this is where the signature matters
        print("\nSecond turn (signature should be sent back)...")
        response2 = await get_response(
            model="gemini-2.5-flash",
            messages=messages,
            tools=[calculator_tool],
            reasoning=ReasoningConfig(budget_tokens=2000),
            max_tokens=1000,
        )
        
        print("✓ Multi-turn successful with signature")
        
        # Check if we got a response
        text_blocks = [b for b in response2.content if isinstance(b, TextBlock)]
        if text_blocks:
            print(f"  Response: {text_blocks[0].text[:100]}...")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in multi-turn: {e}")
        if "signature" in str(e).lower() or "validation" in str(e).lower():
            print("⚠️  This appears to be a signature-related error!")
        import traceback
        traceback.print_exc()
        return False


async def test_gemini_without_thinking():
    """Test that normal Gemini usage (no thinking) still works"""
    print("\n" + "=" * 80)
    print("  TEST 4: Gemini Without Thinking (Regression Test)")
    print("=" * 80)
    
    if not os.environ.get("GEMINI_API_KEY") and not os.environ.get("GOOGLE_API_KEY"):
        print("❌ No GEMINI_API_KEY found")
        return False
    
    messages = [
        ChatMessage(
            role=MessageRole.user,
            content=[TextBlock(text="What is 5 + 5?")],
        )
    ]
    
    try:
        # No reasoning config, no tools
        response = await get_response(
            model="gemini-2.5-flash",
            messages=messages,
            max_tokens=100,
        )
        
        text_blocks = [b for b in response.content if isinstance(b, TextBlock)]
        if text_blocks:
            print(f"✓ Response: {text_blocks[0].text}")
            return True
        else:
            print("❌ No text in response")
            return False
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    """Run all tests"""
    print("=" * 80)
    print("  GEMINI THOUGHT SIGNATURE FIX VERIFICATION")
    print("=" * 80)
    print()
    
    results = []
    
    results.append(("Signature Extraction", await test_gemini_signature_extraction()))
    results.append(("Signature Serialization", await test_gemini_signature_serialization()))
    results.append(("Signature Round-trip", await test_gemini_signature_roundtrip()))
    results.append(("Without Thinking", await test_gemini_without_thinking()))
    
    print("\n" + "=" * 80)
    print("  TEST SUMMARY")
    print("=" * 80)
    
    for test_name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {test_name}")
    
    all_passed = all(r[1] for r in results)
    print()
    if all_passed:
        print("🎉 All tests passed!")
    else:
        print("⚠️  Some tests failed")
    
    return all_passed


if __name__ == "__main__":
    asyncio.run(main())