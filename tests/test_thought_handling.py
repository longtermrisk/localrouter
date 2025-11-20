"""
Test script to verify correct handling of reasoning/thought tokens across OpenAI, Anthropic, and Gemini.

Key behaviors to verify:
1. OpenAI: Hidden reasoning tokens, NOT sent back in history
2. Anthropic: Visible thinking blocks, MUST be sent back in history
3. Gemini: Thought signatures in parts, MUST be sent back exactly as received
"""

import asyncio
import os
import json
from localrouter.dtypes import (
    ChatMessage,
    MessageRole,
    TextBlock,
    ThinkingBlock,
    ToolUseBlock,
    ToolResultBlock,
    ToolDefinition,
    ReasoningConfig,
    genai_format,
)
from localrouter.llm import get_response


# Simple calculator tool for testing
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


def print_section(title):
    """Print a formatted section header"""
    print(f"\n{'=' * 80}")
    print(f"  {title}")
    print('=' * 80)


async def test_openai_reasoning():
    """Test OpenAI reasoning - hidden tokens that should NOT be sent back"""
    print_section("Testing OpenAI Reasoning (o1/o3-mini/GPT-5)")
    
    if not os.environ.get("OPENAI_API_KEY"):
        print("❌ No OPENAI_API_KEY found, skipping test")
        return
    
    # Test with gpt-4o-mini since o1/o3/gpt-5 may not be available
    messages = [
        ChatMessage(
            role=MessageRole.user,
            content=[TextBlock(text="What is 7 * 8?")],
        )
    ]
    
    try:
        response = await get_response(
            model="gpt-4o-mini",
            messages=messages,
            max_tokens=100,
        )
        
        # Check that response doesn't contain thinking blocks
        thinking_blocks = [b for b in response.content if isinstance(b, ThinkingBlock)]
        text_blocks = [b for b in response.content if isinstance(b, TextBlock)]
        
        print(f"✓ Response received")
        print(f"  Thinking blocks: {len(thinking_blocks)} (should be 0 for OpenAI)")
        print(f"  Text blocks: {len(text_blocks)}")
        
        if thinking_blocks:
            print(f"⚠️  WARNING: OpenAI response contains thinking blocks (unexpected)")
        
        # Test multi-turn - thinking blocks should never be in history
        messages.append(response)
        messages.append(ChatMessage(
            role=MessageRole.user,
            content=[TextBlock(text="Now multiply that by 2")],
        ))
        
        # Check the openai_format to ensure no thinking blocks are sent
        openai_formatted = response.openai_format()
        if isinstance(openai_formatted, dict):
            content_str = str(openai_formatted.get("content", ""))
            if "thinking" in content_str.lower():
                print(f"⚠️  WARNING: Thinking content appears in OpenAI format")
        
        response2 = await get_response(
            model="gpt-4o-mini",
            messages=messages,
            max_tokens=100,
        )
        
        print(f"✓ Multi-turn successful")
        
    except Exception as e:
        print(f"❌ Error: {e}")


async def test_anthropic_thinking():
    """Test Anthropic thinking - visible blocks that MUST be sent back"""
    print_section("Testing Anthropic Extended Thinking (Claude Sonnet 3.7/4)")
    
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("❌ No ANTHROPIC_API_KEY found, skipping test")
        return
    
    messages = [
        ChatMessage(
            role=MessageRole.user,
            content=[TextBlock(text="What is 15 * 12? Think step by step.")],
        )
    ]
    
    try:
        response = await get_response(
            model="claude-3-7-sonnet-20250219",
            messages=messages,
            reasoning=ReasoningConfig(budget_tokens=1024),
            max_tokens=2000,
        )
        
        thinking_blocks = [b for b in response.content if isinstance(b, ThinkingBlock)]
        text_blocks = [b for b in response.content if isinstance(b, TextBlock)]
        
        print(f"✓ Response received")
        print(f"  Thinking blocks: {len(thinking_blocks)} (should be > 0)")
        print(f"  Text blocks: {len(text_blocks)}")
        
        if thinking_blocks:
            print(f"  Thinking preview: {thinking_blocks[0].thinking[:100]}...")
            print(f"  Has signature: {thinking_blocks[0].signature is not None}")
        else:
            print(f"⚠️  WARNING: No thinking blocks (expected for extended thinking)")
        
        # Test that thinking blocks are properly formatted for Anthropic
        anthropic_formatted = response.anthropic_format()
        thinking_in_format = [c for c in anthropic_formatted["content"] if c.get("type") == "thinking"]
        print(f"  Thinking blocks in anthropic_format: {len(thinking_in_format)}")
        
        if thinking_in_format:
            first_thinking = thinking_in_format[0]
            print(f"  Format has 'thinking' field: {'thinking' in first_thinking}")
            print(f"  Format has 'signature' field: {'signature' in first_thinking}")
        
        # Test multi-turn - thinking blocks MUST be preserved
        messages.append(response)
        messages.append(ChatMessage(
            role=MessageRole.user,
            content=[TextBlock(text="Now multiply that result by 3")],
        ))
        
        response2 = await get_response(
            model="claude-3-7-sonnet-20250219",
            messages=messages,
            reasoning=ReasoningConfig(budget_tokens=1024),
            max_tokens=2000,
        )
        
        print(f"✓ Multi-turn successful (thinking context preserved)")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


async def test_anthropic_with_tools():
    """Test Anthropic thinking with tool use - critical case from bug report"""
    print_section("Testing Anthropic Thinking + Tool Use (Critical Test)")
    
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("❌ No ANTHROPIC_API_KEY found, skipping test")
        return
    
    messages = [
        ChatMessage(
            role=MessageRole.user,
            content=[TextBlock(text="Calculate 25 * 4 using the calculator tool. Think about it first.")],
        )
    ]
    
    try:
        response = await get_response(
            model="claude-3-7-sonnet-20250219",
            messages=messages,
            tools=[calculator_tool],
            reasoning=ReasoningConfig(budget_tokens=1024),
            max_tokens=3000,
        )
        
        thinking_blocks = [b for b in response.content if isinstance(b, ThinkingBlock)]
        tool_use_blocks = [b for b in response.content if isinstance(b, ToolUseBlock)]
        
        print(f"✓ Response received")
        print(f"  Thinking blocks: {len(thinking_blocks)}")
        print(f"  Tool use blocks: {len(tool_use_blocks)}")
        
        if tool_use_blocks:
            # Simulate tool execution
            tool_block = tool_use_blocks[0]
            expression = tool_block.input.get("expression", "")
            result = eval(expression)
            
            print(f"  Tool called: {tool_block.name}({expression}) = {result}")
            
            # Add tool result and continue
            messages.append(response)
            messages.append(ChatMessage(
                role=MessageRole.user,
                content=[
                    ToolResultBlock(
                        tool_use_id=tool_block.id,
                        content=[TextBlock(text=f"Result: {result}")],
                    )
                ],
            ))
            
            # This is where the original bug would occur
            response2 = await get_response(
                model="claude-3-7-sonnet-20250219",
                messages=messages,
                tools=[calculator_tool],
                reasoning=ReasoningConfig(budget_tokens=1024),
                max_tokens=3000,
            )
            
            print(f"✓ Multi-turn with tool use successful (thinking preserved)")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


async def test_gemini_thinking():
    """Test Gemini thinking - thought signatures that MUST be sent back"""
    print_section("Testing Gemini Thinking + Thought Signatures")
    
    if not os.environ.get("GEMINI_API_KEY") and not os.environ.get("GOOGLE_API_KEY"):
        print("❌ No GEMINI_API_KEY or GOOGLE_API_KEY found, skipping test")
        return
    
    messages = [
        ChatMessage(
            role=MessageRole.user,
            content=[TextBlock(text="What is 9 * 7?")],
        )
    ]
    
    try:
        # Test basic thinking
        response = await get_response(
            model="gemini-2.5-flash",
            messages=messages,
            reasoning=ReasoningConfig(budget_tokens=2000),
            max_tokens=500,
        )
        
        thinking_blocks = [b for b in response.content if isinstance(b, ThinkingBlock)]
        text_blocks = [b for b in response.content if isinstance(b, TextBlock)]
        
        print(f"✓ Response received (no tools)")
        print(f"  Thinking blocks: {len(thinking_blocks)}")
        print(f"  Text blocks: {len(text_blocks)}")
        
        if thinking_blocks:
            print(f"  Thinking preview: {thinking_blocks[0].thinking[:100]}...")
            print(f"  Has signature: {thinking_blocks[0].signature is not None}")
            
    except Exception as e:
        print(f"❌ Error (no tools): {e}")
        import traceback
        traceback.print_exc()


async def test_gemini_thinking_with_tools():
    """Test Gemini thinking with tools - where thought signatures appear"""
    print_section("Testing Gemini Thinking + Tools (Signature Test)")
    
    if not os.environ.get("GEMINI_API_KEY") and not os.environ.get("GOOGLE_API_KEY"):
        print("❌ No GEMINI_API_KEY or GOOGLE_API_KEY found, skipping test")
        return
    
    messages = [
        ChatMessage(
            role=MessageRole.user,
            content=[TextBlock(text="Calculate 15 * 8 using the calculator tool")],
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
        
        thinking_blocks = [b for b in response.content if isinstance(b, ThinkingBlock)]
        tool_use_blocks = [b for b in response.content if isinstance(b, ToolUseBlock)]
        text_blocks = [b for b in response.content if isinstance(b, TextBlock)]
        
        print(f"✓ Response received (with tools)")
        print(f"  Thinking blocks: {len(thinking_blocks)}")
        print(f"  Tool use blocks: {len(tool_use_blocks)}")
        print(f"  Text blocks: {len(text_blocks)}")
        
        if thinking_blocks:
            print(f"  Has signature in thinking: {thinking_blocks[0].signature is not None}")
        
        # Check raw response for signatures
        print(f"\n  Checking for thought signatures in response...")
        
        # Now test multi-turn with tool result
        if tool_use_blocks:
            tool_block = tool_use_blocks[0]
            expression = tool_block.input.get("expression", "")
            result = eval(expression)
            
            print(f"  Tool called: {tool_block.name}({expression}) = {result}")
            
            messages.append(response)
            messages.append(ChatMessage(
                role=MessageRole.user,
                content=[
                    ToolResultBlock(
                        tool_use_id=tool_block.id,
                        content=[TextBlock(text=f"Result: {result}")],
                    )
                ],
            ))
            
            # Check what's being sent back to Gemini
            print(f"\n  Checking genai_format for signatures...")
            formatted = genai_format(messages, [calculator_tool])
            print(f"  Contents in request: {len(formatted['contents'])}")
            
            # Look at the parts in each content
            for i, content in enumerate(formatted['contents']):
                if hasattr(content, 'parts'):
                    print(f"  Content {i} has {len(content.parts)} parts")
                    for j, part in enumerate(content.parts):
                        part_info = str(type(part).__name__)
                        # Check if part has signature
                        if hasattr(part, 'thought') and hasattr(part, 'signature'):
                            part_info += " (has thought+signature)"
                        print(f"    Part {j}: {part_info}")
            
            # Try the second turn
            try:
                response2 = await get_response(
                    model="gemini-2.5-flash",
                    messages=messages,
                    tools=[calculator_tool],
                    reasoning=ReasoningConfig(budget_tokens=2000),
                    max_tokens=1000,
                )
                
                print(f"✓ Multi-turn with tools successful")
                
            except Exception as e2:
                print(f"❌ Multi-turn error: {e2}")
                if "signature" in str(e2).lower() or "validation" in str(e2).lower():
                    print(f"⚠️  THIS LOOKS LIKE A SIGNATURE HANDLING BUG!")
                import traceback
                traceback.print_exc()
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


async def test_gemini_signature_detection():
    """Test if we can detect thought signatures in Gemini responses"""
    print_section("Testing Gemini Signature Detection")
    
    if not os.environ.get("GEMINI_API_KEY") and not os.environ.get("GOOGLE_API_KEY"):
        print("❌ No GEMINI_API_KEY or GOOGLE_API_KEY found, skipping test")
        return
    
    try:
        from google import genai
        from google.genai import types as genai_types
        
        api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        client = genai.Client(api_key=api_key)
        
        # Make a request with thinking + tools
        config = genai_types.GenerateContentConfig(
            thinking_config=genai_types.ThinkingConfig(
                thinking_budget=2000,
                include_thoughts=True,
            ),
            tools=[
                genai_types.Tool(
                    function_declarations=[
                        genai_types.FunctionDeclaration(
                            name="calculator",
                            description="Calculate expressions",
                            parameters_json_schema={
                                "type": "object",
                                "properties": {
                                    "expression": {"type": "string"}
                                },
                                "required": ["expression"]
                            }
                        )
                    ]
                )
            ],
        )
        
        response = await client.aio.models.generate_content(
            model="gemini-2.5-flash",
            contents=[genai_types.UserContent(parts=[
                genai_types.Part.from_text(text="Calculate 20 * 5")
            ])],
            config=config,
        )
        
        print(f"✓ Raw Gemini response received")
        
        # Inspect the response structure
        if hasattr(response, 'candidates') and response.candidates:
            candidate = response.candidates[0]
            if hasattr(candidate, 'content') and hasattr(candidate.content, 'parts'):
                print(f"  Parts in response: {len(candidate.content.parts)}")
                
                for i, part in enumerate(candidate.content.parts):
                    print(f"\n  Part {i}:")
                    print(f"    Type: {type(part).__name__}")
                    
                    # Check all attributes
                    attrs = dir(part)
                    interesting_attrs = [a for a in attrs if not a.startswith('_') and a in [
                        'text', 'thought', 'signature', 'function_call', 'function_response'
                    ]]
                    
                    for attr in interesting_attrs:
                        try:
                            value = getattr(part, attr, None)
                            if value:
                                if attr in ['text', 'thought']:
                                    print(f"    {attr}: {str(value)[:50]}...")
                                elif attr == 'signature':
                                    print(f"    signature: {str(value)[:100]}... (length: {len(str(value))})")
                                else:
                                    print(f"    {attr}: {value}")
                        except:
                            pass
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


async def main():
    """Run all thought handling tests"""
    print("=" * 80)
    print("  COMPREHENSIVE THOUGHT TOKEN HANDLING TEST")
    print("=" * 80)
    print("\nThis test verifies correct handling of reasoning/thought tokens across:")
    print("  1. OpenAI: Hidden reasoning (not sent back)")
    print("  2. Anthropic: Visible thinking blocks (must be sent back)")
    print("  3. Gemini: Thought signatures (must be sent back exactly)")
    print()
    
    await test_openai_reasoning()
    await test_anthropic_thinking()
    await test_anthropic_with_tools()
    await test_gemini_thinking()
    await test_gemini_thinking_with_tools()
    await test_gemini_signature_detection()
    
    print_section("Test Summary")
    print("Review the output above for any ❌ errors or ⚠️ warnings.")
    print("Focus especially on Gemini signature handling in multi-turn with tools.")


if __name__ == "__main__":
    asyncio.run(main())