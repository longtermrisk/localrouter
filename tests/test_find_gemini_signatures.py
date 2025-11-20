"""
Deep inspection to find exactly where Gemini stores thought signatures.
"""

import asyncio
import os
import json
import pickle
from google import genai
from google.genai import types as genai_types


def inspect_object(obj, name="object", depth=0, max_depth=3):
    """Recursively inspect an object to find all attributes"""
    if depth > max_depth:
        return
    
    indent = "  " * depth
    print(f"{indent}{name}: {type(obj).__name__}")
    
    if hasattr(obj, '__dict__'):
        for key, value in obj.__dict__.items():
            if key.startswith('_'):
                continue
            if value is None:
                continue
            if callable(value):
                continue
            
            if isinstance(value, (str, int, float, bool)):
                if isinstance(value, str) and len(value) > 100:
                    print(f"{indent}  {key}: {value[:100]}... (len={len(value)})")
                else:
                    print(f"{indent}  {key}: {value}")
            elif isinstance(value, dict):
                print(f"{indent}  {key}: dict with keys: {list(value.keys())}")
                for k, v in value.items():
                    if isinstance(v, (str, int, float, bool)):
                        print(f"{indent}    {k}: {v}")
            elif isinstance(value, (list, tuple)):
                print(f"{indent}  {key}: {type(value).__name__} of length {len(value)}")
                if len(value) > 0 and depth < max_depth - 1:
                    inspect_object(value[0], f"{key}[0]", depth + 1, max_depth)
            else:
                inspect_object(value, key, depth + 1, max_depth)


async def test_signature_location():
    """Find where signatures are stored in Gemini responses"""
    print("=" * 80)
    print("  SEARCHING FOR GEMINI THOUGHT SIGNATURES")
    print("=" * 80)
    
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        print("❌ No API key found")
        return
    
    client = genai.Client(api_key=api_key)
    
    # Test with thinking + tools (where signatures should appear)
    print("\n1. Making request with thinking + tools")
    print("-" * 80)
    
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
            genai_types.Part.from_text(text="Calculate 13 * 7 using the calculator")
        ])],
        config=config,
    )
    
    print("\n2. Inspecting response structure")
    print("-" * 80)
    
    # Inspect the full response
    inspect_object(response, "response", max_depth=4)
    
    print("\n\n3. Detailed Part inspection")
    print("-" * 80)
    
    if hasattr(response, 'candidates') and response.candidates:
        candidate = response.candidates[0]
        if hasattr(candidate, 'content') and hasattr(candidate.content, 'parts'):
            for i, part in enumerate(candidate.content.parts):
                print(f"\n--- Part {i} ---")
                print(f"Type: {type(part)}")
                print(f"Dir: {[a for a in dir(part) if not a.startswith('_')]}")
                
                # Check every attribute
                for attr in dir(part):
                    if attr.startswith('_'):
                        continue
                    try:
                        value = getattr(part, attr)
                        if value is None or callable(value):
                            continue
                        
                        attr_name = attr
                        if isinstance(value, str):
                            if len(value) > 200:
                                print(f"  {attr_name}: '{value[:100]}...' (length: {len(value)})")
                            else:
                                print(f"  {attr_name}: '{value}'")
                        elif isinstance(value, bool):
                            print(f"  {attr_name}: {value}")
                        elif isinstance(value, (int, float)):
                            print(f"  {attr_name}: {value}")
                        elif hasattr(value, '__dict__'):
                            print(f"  {attr_name}: <{type(value).__name__}>")
                            # Check if it has a signature attribute
                            if hasattr(value, 'signature'):
                                sig = getattr(value, 'signature')
                                if sig:
                                    print(f"    → signature: {sig[:100] if isinstance(sig, str) else sig}")
                            # Print non-private attributes
                            for subattr in dir(value):
                                if not subattr.startswith('_'):
                                    try:
                                        subval = getattr(value, subattr)
                                        if subval is not None and not callable(subval):
                                            if isinstance(subval, str) and len(subval) > 50:
                                                print(f"    .{subattr}: '{subval[:50]}...'")
                                            elif isinstance(subval, (str, int, float, bool)):
                                                print(f"    .{subattr}: {subval}")
                                    except:
                                        pass
                        else:
                            print(f"  {attr_name}: {type(value).__name__}")
                    except Exception as e:
                        pass
    
    print("\n\n4. Testing model_dump() for serialization")
    print("-" * 80)
    
    if hasattr(response, 'candidates') and response.candidates:
        candidate = response.candidates[0]
        if hasattr(candidate, 'content') and hasattr(candidate.content, 'parts'):
            for i, part in enumerate(candidate.content.parts):
                print(f"\n--- Part {i} model_dump() ---")
                try:
                    if hasattr(part, 'model_dump'):
                        dumped = part.model_dump()
                        print(json.dumps(dumped, indent=2, default=str)[:1000])
                        
                        # Check for signature keys
                        if 'signature' in dumped:
                            print(f"\n⭐ FOUND 'signature' in model_dump()!")
                            print(f"   Type: {type(dumped['signature'])}")
                            print(f"   Value: {dumped['signature']}")
                        
                        # Check all keys
                        print(f"\nKeys in dump: {list(dumped.keys())}")
                        for key in dumped.keys():
                            if 'sig' in key.lower():
                                print(f"⭐ FOUND key with 'sig': {key} = {dumped[key]}")
                except Exception as e:
                    print(f"Error: {e}")
    
    print("\n\n5. Testing dict() method")
    print("-" * 80)
    
    if hasattr(response, 'candidates') and response.candidates:
        candidate = response.candidates[0]
        if hasattr(candidate, 'content') and hasattr(candidate.content, 'parts'):
            for i, part in enumerate(candidate.content.parts):
                print(f"\n--- Part {i} dict() ---")
                try:
                    # Try deprecated dict() method
                    if hasattr(part, 'dict'):
                        part_dict = part.dict()
                        print(json.dumps(part_dict, indent=2, default=str)[:1000])
                        
                        if 'signature' in part_dict:
                            print(f"\n⭐ FOUND 'signature' in dict()!")
                            print(f"   Value: {part_dict['signature']}")
                except Exception as e:
                    print(f"Error (expected if deprecated): {e}")
    
    print("\n\n6. Checking if Part has __dict__ directly")
    print("-" * 80)
    
    if hasattr(response, 'candidates') and response.candidates:
        candidate = response.candidates[0]
        if hasattr(candidate, 'content') and hasattr(candidate.content, 'parts'):
            for i, part in enumerate(candidate.content.parts):
                print(f"\n--- Part {i} __dict__ ---")
                if hasattr(part, '__dict__'):
                    for key, value in part.__dict__.items():
                        if value is None or callable(value):
                            continue
                        print(f"  {key}: {type(value).__name__}", end="")
                        if isinstance(value, str) and len(value) > 50:
                            print(f" = '{value[:50]}...'")
                        elif isinstance(value, (str, int, float, bool)):
                            print(f" = {value}")
                        else:
                            print()
                            # Check sub-attributes
                            if hasattr(value, '__dict__'):
                                for subkey, subval in value.__dict__.items():
                                    if subval is not None and not callable(subval):
                                        print(f"    .{subkey}: {type(subval).__name__}", end="")
                                        if isinstance(subval, str) and 'sig' in subkey.lower():
                                            print(f" ⭐ = {subval[:100] if len(subval) > 100 else subval}")
                                        elif isinstance(subval, (str, int, float, bool)):
                                            print(f" = {subval}")
                                        else:
                                            print()
    
    print("\n\n7. Test serialization and reconstruction")
    print("-" * 80)
    
    if hasattr(response, 'candidates') and response.candidates:
        candidate = response.candidates[0]
        if hasattr(candidate, 'content') and hasattr(candidate.content, 'parts'):
            original_parts = candidate.content.parts
            
            print("Original parts:")
            for i, part in enumerate(original_parts):
                print(f"  Part {i}: {type(part).__name__}")
            
            # Try to serialize using model_dump
            print("\nSerializing with model_dump()...")
            serialized_parts = []
            for part in original_parts:
                if hasattr(part, 'model_dump'):
                    serialized_parts.append(part.model_dump())
            
            print(f"Serialized {len(serialized_parts)} parts")
            print(f"JSON length: {len(json.dumps(serialized_parts, default=str))}")
            
            # Try to reconstruct
            print("\nAttempting to reconstruct Parts...")
            try:
                reconstructed_parts = []
                for i, serialized in enumerate(serialized_parts):
                    print(f"  Part {i} keys: {list(serialized.keys())}")
                    
                    # Check what Part class expects
                    if hasattr(genai_types.Part, 'model_validate'):
                        reconstructed = genai_types.Part.model_validate(serialized)
                        reconstructed_parts.append(reconstructed)
                        print(f"  → Successfully reconstructed")
                    else:
                        print(f"  → No model_validate method")
                
                if reconstructed_parts:
                    print(f"\n✓ Successfully reconstructed {len(reconstructed_parts)} parts")
                    
                    # Compare
                    print("\nComparing original vs reconstructed:")
                    for i, (orig, recon) in enumerate(zip(original_parts, reconstructed_parts)):
                        print(f"  Part {i}:")
                        if hasattr(orig, 'model_dump') and hasattr(recon, 'model_dump'):
                            orig_dump = orig.model_dump()
                            recon_dump = recon.model_dump()
                            if orig_dump == recon_dump:
                                print(f"    ✓ Identical")
                            else:
                                print(f"    ✗ Different")
                                # Find differences
                                all_keys = set(orig_dump.keys()) | set(recon_dump.keys())
                                for key in all_keys:
                                    if orig_dump.get(key) != recon_dump.get(key):
                                        print(f"      {key}: {orig_dump.get(key)} != {recon_dump.get(key)}")
            except Exception as e:
                print(f"Reconstruction error: {e}")
                import traceback
                traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(test_signature_location())