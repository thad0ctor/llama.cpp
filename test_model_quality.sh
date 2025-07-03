#!/bin/bash

# Quality Test Script for Qwen3-235B Model
# Compares output between quantized and unquantized KV cache

set -e

echo "=== Model Quality Test Script ==="
echo "This script will test both configurations and compare outputs"
echo

# Test prompt - something that requires coherent reasoning
TEST_PROMPT="Explain the concept of artificial intelligence in simple terms, covering its benefits and potential risks."

# Function to test a server
test_server() {
    local port=$1
    local config_name=$2
    
    echo "Testing $config_name (port $port)..."
    
    # Check if server is running
    if ! curl -s "http://localhost:$port/health" > /dev/null 2>&1; then
        echo "❌ Server not running on port $port"
        return 1
    fi
    
    echo "✓ Server is running"
    
    # Send test prompt
    local response=$(curl -s -X POST "http://localhost:$port/completion" \
        -H "Content-Type: application/json" \
        -d "{
            \"prompt\": \"$TEST_PROMPT\",
            \"n_predict\": 200,
            \"temperature\": 0.7,
            \"top_p\": 0.95,
            \"stream\": false
        }" | jq -r '.content' 2>/dev/null)
    
    if [ $? -eq 0 ] && [ -n "$response" ]; then
        echo "✓ Got response from $config_name"
        echo
        echo "--- $config_name Response ---"
        echo "$response"
        echo
        echo "--- End $config_name Response ---"
        echo
        
        # Basic quality checks
        local word_count=$(echo "$response" | wc -w)
        local char_count=$(echo "$response" | wc -c)
        
        echo "📊 $config_name Stats:"
        echo "  - Word count: $word_count"
        echo "  - Character count: $char_count"
        
        # Check for obvious gibberish patterns
        local gibberish_patterns=0
        if echo "$response" | grep -q "[a-zA-Z]\{20,\}"; then
            ((gibberish_patterns++))
            echo "  - ⚠️  Contains very long words (possible gibberish)"
        fi
        
        if echo "$response" | grep -q "[^a-zA-Z0-9 .,!?'\"\n\r\t-]"; then
            ((gibberish_patterns++))
            echo "  - ⚠️  Contains unusual characters"
        fi
        
        if [ $gibberish_patterns -eq 0 ]; then
            echo "  - ✅ No obvious gibberish patterns detected"
        fi
        
        return 0
    else
        echo "❌ Failed to get response from $config_name"
        return 1
    fi
}

echo "🧪 Starting quality comparison test..."
echo "Prompt: $TEST_PROMPT"
echo
echo "========================================"
echo

# Test the quantized version (port 5001)
test_server 5001 "Quantized KV Cache (INT8)"
echo

# Test the unquantized version (port 5002) 
test_server 5002 "Unquantized KV Cache"
echo

# Test the INT4 version (port 5003) if running
test_server 5003 "INT4 Quantized KV Cache (K=INT4, V=INT8)"
echo

echo "========================================"
echo "✅ Quality test completed!"
echo
echo "📋 Instructions:"
echo "1. Compare the responses above"
echo "2. Look for coherence, relevance, and absence of gibberish"
echo "3. Quality ranking (expected): Unquantized > INT8 > INT4"
echo "4. If INT4 produces gibberish, it's too aggressive for this model"
echo "5. If all versions have issues, the problem is elsewhere in the optimization stack"
echo 