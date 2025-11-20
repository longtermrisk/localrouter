"""
Detailed test to understand Gemini thought signatures.
"""

import asyncio
import os
from google import genai
from google.genai import types as genai_types


async def test_gemini_signature_structure():
    """Deep dive into Gemini response structure to find signatures"""
    print("=" * 80)
    print("  DETAILED GEMINI SIGNATURE INSPECTION")
    print("=" * 80)
    
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        print("❌ No API key found")
        return
    
    client = genai.Client(api_key=api_key)
    
    # Test 1: With thinking but NO tools (should NOT have signatures)
    print("\n1. Testing: Thinking WITHOUT tools (no signatures expected)")
    print("-" * 80)
    
    config1 = genai_types.GenerateContentConfig(
        thinking_config=genai_types.ThinkingConfig(
            thinking_budget=2000,
            include_thoughts=True,
        ),
    )
    
    response1 = await client.aio.models.generate_content(
        model="gemini-2.5-flash",
        contents=[genai_types.UserContent(parts=[
            genai_types.Part.from_text(text="What is 7 * 8?")
        ])],
        config=config1,
    )
    
    print(f"✓ Response received")
    if hasattr(response1, 'candidates') and response1.candidates:
        candidate = response1.candidates[0]
        if hasattr(candidate, 'content') and hasattr(candidate.content, 'parts'):
            print(f"  Parts: {len(candidate.content.parts)}")
            
            for i, part in enumerate(candidate.content.parts):
                print(f"\n  Part {i}:")
                # Get all non-private attributes
                attrs = [a for a in dir(part) if not a.startswith('_')]
                for attr in attrs:
                    try:
                        value = getattr(part, attr)
                        if value and not callable(value):
                            if isinstance(value, str):
                                if len(value) > 100:
                                    print(f"    {attr}: {value[:100]}... (length: {len(value)})")
                                else:
                                    print(f"    {attr}: {value}")
                            elif isinstance(value, bool):
                                print(f"    {attr}: {value}")
                            elif hasattr(value, '__dict__'):
                                print(f"    {attr}: <object with attrs: {list(value.__dict__.keys())}>")
                    except:
                        pass
    
    # Test 2: With thinking AND tools (SHOULD have signatures)
    print("\n\n2. Testing: Thinking WITH tools (signatures expected)")
    print("-" * 80)
    
    config2 = genai_types.GenerateContentConfig(
        thinking_config=genai_types.ThinkingConfig(
            thinking_budget=2000,
            include_thoughts=True,
        ),
        tools=[
            genai_types.Tool(
                function_declarations=[
                    genai_types.FunctionDeclaration(
                        name="calculator",
                        description="Calculate mathematical expressions",
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
    
    response2 = await client.aio.models.generate_content(
        model="gemini-2.5-flash",
        contents=[genai_types.UserContent(parts=[
            genai_types.Part.from_text(text="Calculate 12 * 9")
        ])],
        config=config2,
    )
    
    print(f"✓ Response received")
    if hasattr(response2, 'candidates') and response2.candidates:
        candidate = response2.candidates[0]
        if hasattr(candidate, 'content') and hasattr(candidate.content, 'parts'):
            print(f"  Parts: {len(candidate.content.parts)}")
            
            for i, part in enumerate(candidate.content.parts):
                print(f"\n  Part {i}:")
                # Get all non-private attributes
                attrs = [a for a in dir(part) if not a.startswith('_')]
                for attr in attrs:
                    try:
                        value = getattr(part, attr)
                        if value and not callable(value):
                            if isinstance(value, str):
                                if len(value) > 100:
                                    print(f"    {attr}: {value[:100]}... (length: {len(value)})")
                                else:
                                    print(f"    {attr}: {value}")
                            elif isinstance(value, bool):
                                print(f"    {attr}: {value}")
                            elif hasattr(value, '__dict__'):
                                # Print object structure
                                obj_attrs = [a for a in dir(value) if not a.startswith('_')]
                                print(f"    {attr}: <{type(value).__name__}>")
                                for obj_attr in obj_attrs[:10]:  # Limit to first 10
                                    try:
                                        obj_val = getattr(value, obj_attr)
                                        if obj_val and not callable(obj_val):
                                            if isinstance(obj_val, str):
                                                print(f"      .{obj_attr}: {obj_val[:50]}...")
                                            else:
                                                print(f"      .{obj_attr}: {obj_val}")
                                    except:
                                        pass
                    except:
                        pass
    
    # Test 3: Check raw response object structure
    print("\n\n3. Checking raw response2 object structure")
    print("-" * 80)
    
    print(f"Response type: {type(response2).__name__}")
    response_attrs = [a for a in dir(response2) if not a.startswith('_')]
    print(f"Response attributes: {response_attrs[:20]}")
    
    # Check candidates
    if hasattr(response2, 'candidates'):
        print(f"\nCandidates: {len(response2.candidates)}")
        for i, cand in enumerate(response2.candidates):
            print(f"  Candidate {i}:")
            cand_attrs = [a for a in dir(cand) if not a.startswith('_')]
            print(f"    Attributes: {cand_attrs[:15]}")
            
            # Check content
            if hasattr(cand, 'content'):
                content = cand.content
                print(f"    Content type: {type(content).__name__}")
                content_attrs = [a for a in dir(content) if not a.startswith('_')]
                print(f"    Content attributes: {content_attrs[:15]}")
                
                # Check parts
                if hasattr(content, 'parts'):
                    print(f"    Parts: {len(content.parts)}")
    
    # Test 4: Try to serialize response to JSON to see all fields
    print("\n\n4. Attempting to serialize response (may show hidden fields)")
    print("-" * 80)
    
    try:
        # Try to_dict if available
        if hasattr(response2, 'to_dict'):
            import json
            response_dict = response2.to_dict()
            print(json.dumps(response_dict, indent=2, default=str)[:1000])
    except Exception as e:
        print(f"Serialization error: {e}")
    
    # Test 5: Multi-turn to see if we lose context
    print("\n\n5. Testing multi-turn with tool result")
    print("-" * 80)
    
    # Simulate tool execution
    first_response_parts = response2.candidates[0].content.parts
    has_function_call = any(hasattr(p, 'function_call') and p.function_call for p in first_response_parts)
    
    if has_function_call:
        print("First response had function call, simulating execution...")
        
        # Build conversation with tool result
        # The key question: Are we passing back the entire Part objects with their signatures?
        
        # According to docs: "Return the entire response with all parts back to the model"
        # Let's test both approaches
        
        # Approach 1: Send back entire Parts
        print("\nApproach 1: Sending back ENTIRE response parts")
        contents_v1 = [
            genai_types.UserContent(parts=[
                genai_types.Part.from_text(text="Calculate 12 * 9")
            ]),
            genai_types.ModelContent(parts=first_response_parts),  # Send ENTIRE parts
            genai_types.UserContent(parts=[
                genai_types.Part.from_text(text="The result is 108")
            ]),
        ]
        
        try:
            response_v1 = await client.aio.models.generate_content(
                model="gemini-2.5-flash",
                contents=contents_v1,
                config=config2,
            )
            print("✓ Approach 1 successful")
        except Exception as e:
            print(f"❌ Approach 1 failed: {e}")
        
        # Approach 2: Manually reconstruct parts (what we're currently doing)
        print("\nApproach 2: Reconstructing parts from extracted data")
        
        reconstructed_parts = []
        for part in first_response_parts:
            if hasattr(part, 'text') and part.text:
                reconstructed_parts.append(genai_types.Part.from_text(text=part.text))
            elif hasattr(part, 'function_call') and part.function_call:
                reconstructed_parts.append(genai_types.Part.from_function_call(
                    name=part.function_call.name,
                    args=dict(part.function_call.args)
                ))
        
        contents_v2 = [
            genai_types.UserContent(parts=[
                genai_types.Part.from_text(text="Calculate 12 * 9")
            ]),
            genai_types.ModelContent(parts=reconstructed_parts),  # Send reconstructed
            genai_types.UserContent(parts=[
                genai_types.Part.from_text(text="The result is 108")
            ]),
        ]
        
        try:
            response_v2 = await client.aio.models.generate_content(
                model="gemini-2.5-flash",
                contents=contents_v2,
                config=config2,
            )
            print("✓ Approach 2 successful")
        except Exception as e:
            print(f"❌ Approach 2 failed: {e}")
            print(f"   Error details: {str(e)}")


if __name__ == "__main__":
    asyncio.run(test_gemini_signature_structure())