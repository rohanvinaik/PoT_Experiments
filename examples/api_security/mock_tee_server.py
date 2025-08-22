#!/usr/bin/env python3
"""
Mock TEE Server for Testing

Simulates a TEE-enabled API server that provides model inference
with attestation support for testing the PoT framework.
"""

import json
import time
import uuid
import hashlib
import argparse
from flask import Flask, request, jsonify
from typing import Dict, Any, Optional
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.pot.security.tee_attestation import (
    AttestationType,
    ModelIdentity,
    create_attestation_provider
)
from src.pot.api.secure_binding import (
    APITranscript,
    SecureAPIBinder,
    BindingPolicy
)


app = Flask(__name__)

# Global state
MODEL_IDENTITY = None
ATTESTATION_PROVIDER = None
API_BINDER = None
SESSION_ATTESTATIONS = {}


def initialize_server(provider_type: AttestationType, model_name: str, model_version: str):
    """Initialize mock server with TEE provider"""
    global MODEL_IDENTITY, ATTESTATION_PROVIDER, API_BINDER
    
    # Create model identity
    MODEL_IDENTITY = ModelIdentity(
        model_hash=hashlib.sha256(f"{model_name}_{model_version}".encode()).hexdigest(),
        model_name=model_name,
        version=model_version,
        provider="mock_server",
        architecture="transformer",
        parameter_count=175000000,  # Mock 175M params
        metadata={
            'server_version': '1.0.0',
            'deployment_id': str(uuid.uuid4())
        }
    )
    
    # Create attestation provider
    config = {
        'mock_id': str(uuid.uuid4()),
        'enclave_id': str(uuid.uuid4()),
        'vm_id': str(uuid.uuid4())
    }
    
    ATTESTATION_PROVIDER = create_attestation_provider(provider_type, config)
    
    # Create API binder with relaxed policy for testing
    API_BINDER = SecureAPIBinder(
        policy=BindingPolicy.relaxed_policy()
    )
    
    print(f"Server initialized with {provider_type.value} attestation")
    print(f"Model: {model_name} v{model_version}")
    print(f"Model Hash: {MODEL_IDENTITY.model_hash}")


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'provider': ATTESTATION_PROVIDER.provider_type.value if ATTESTATION_PROVIDER else 'none',
        'model': MODEL_IDENTITY.model_name if MODEL_IDENTITY else 'none'
    })


@app.route('/attestation/create', methods=['POST'])
def create_attestation():
    """Create new attestation for session"""
    data = request.json or {}
    nonce = data.get('nonce', str(uuid.uuid4()))
    
    if not ATTESTATION_PROVIDER or not MODEL_IDENTITY:
        return jsonify({'error': 'Server not initialized'}), 500
    
    # Generate attestation
    attestation = ATTESTATION_PROVIDER.generate_attestation(
        MODEL_IDENTITY,
        nonce,
        additional_data={'session_id': str(uuid.uuid4())}
    )
    
    # Store for session
    session_id = str(uuid.uuid4())
    SESSION_ATTESTATIONS[session_id] = attestation
    
    return jsonify({
        'session_id': session_id,
        'attestation': attestation.to_dict(),
        'model_identity': {
            'hash': MODEL_IDENTITY.compute_identity_hash(),
            'name': MODEL_IDENTITY.model_name,
            'version': MODEL_IDENTITY.version
        }
    })


@app.route('/inference', methods=['POST'])
def inference():
    """Perform inference with attestation binding"""
    data = request.json or {}
    
    # Get session attestation
    session_id = data.get('session_id')
    if not session_id or session_id not in SESSION_ATTESTATIONS:
        return jsonify({'error': 'Invalid session. Create attestation first.'}), 401
    
    attestation = SESSION_ATTESTATIONS[session_id]
    
    # Extract inference parameters
    prompt = data.get('prompt', '')
    max_tokens = data.get('max_tokens', 50)
    temperature = data.get('temperature', 0.7)
    
    # Simulate inference
    start_time = time.time()
    
    # Mock response generation
    response_text = f"Mock response to: '{prompt[:50]}...' "
    response_text += f"[Generated with {MODEL_IDENTITY.model_name} "
    response_text += f"using {ATTESTATION_PROVIDER.provider_type.value} attestation]"
    
    inference_time = (time.time() - start_time) * 1000  # ms
    
    # Create API transcript
    transcript = APITranscript(
        transcript_id=str(uuid.uuid4()),
        timestamp=time.time(),
        endpoint='/inference',
        method='POST',
        request={
            'prompt': prompt,
            'max_tokens': max_tokens,
            'temperature': temperature
        },
        response={
            'text': response_text,
            'tokens_generated': len(response_text.split()),
            'model': MODEL_IDENTITY.model_name
        },
        latency_ms=inference_time,
        metadata={
            'session_id': session_id,
            'attestation_type': attestation.provider_type.value
        }
    )
    
    # Bind transcript to attestation
    bound_transcript = API_BINDER.bind_transcript(
        transcript,
        MODEL_IDENTITY,
        attestation,
        verify_immediately=False  # Skip verification for mock
    )
    
    return jsonify({
        'transcript_id': transcript.transcript_id,
        'response': response_text,
        'inference_time_ms': inference_time,
        'binding': {
            'signature': bound_transcript.binding_signature,
            'nonce': bound_transcript.binding_nonce,
            'timestamp': bound_transcript.binding_timestamp
        },
        'attestation_hash': hashlib.sha256(
            json.dumps(attestation.to_dict(), sort_keys=True).encode()
        ).hexdigest()
    })


@app.route('/evidence/bundle', methods=['GET'])
def get_evidence_bundle():
    """Get evidence bundle for all transcripts"""
    transcript_ids = list(API_BINDER.bindings.keys())
    
    if not transcript_ids:
        return jsonify({'error': 'No transcripts available'}), 404
    
    bundle = API_BINDER.create_evidence_bundle(
        transcript_ids,
        include_full_transcripts=True
    )
    
    return jsonify(bundle)


@app.route('/evidence/verify', methods=['POST'])
def verify_evidence():
    """Verify evidence bundle"""
    bundle = request.json
    
    if not bundle:
        return jsonify({'error': 'No bundle provided'}), 400
    
    valid, errors = API_BINDER.validate_evidence_bundle(bundle)
    
    return jsonify({
        'valid': valid,
        'errors': errors,
        'timestamp': time.time()
    })


@app.route('/stats', methods=['GET'])
def get_statistics():
    """Get server statistics"""
    stats = API_BINDER.get_statistics() if API_BINDER else {}
    
    stats.update({
        'server': {
            'model': MODEL_IDENTITY.model_name if MODEL_IDENTITY else None,
            'provider': ATTESTATION_PROVIDER.provider_type.value if ATTESTATION_PROVIDER else None,
            'active_sessions': len(SESSION_ATTESTATIONS),
            'total_transcripts': len(API_BINDER.bindings) if API_BINDER else 0
        }
    })
    
    return jsonify(stats)


@app.route('/platform/info', methods=['GET'])
def platform_info():
    """Get platform information"""
    if not ATTESTATION_PROVIDER:
        return jsonify({'error': 'Server not initialized'}), 500
    
    return jsonify(ATTESTATION_PROVIDER.get_platform_info())


def run_mock_server():
    """Run the mock TEE server"""
    parser = argparse.ArgumentParser(description='Mock TEE Server')
    parser.add_argument(
        '--provider',
        type=str,
        choices=['sgx', 'sev', 'nitro', 'vendor', 'mock'],
        default='mock',
        help='TEE provider type'
    )
    parser.add_argument(
        '--model-name',
        type=str,
        default='mock-gpt-2',
        help='Model name'
    )
    parser.add_argument(
        '--model-version',
        type=str,
        default='1.0.0',
        help='Model version'
    )
    parser.add_argument(
        '--port',
        type=int,
        default=8080,
        help='Server port'
    )
    parser.add_argument(
        '--host',
        type=str,
        default='127.0.0.1',
        help='Server host'
    )
    
    args = parser.parse_args()
    
    # Initialize server
    provider_type = AttestationType(args.provider)
    initialize_server(provider_type, args.model_name, args.model_version)
    
    # Run Flask app
    print(f"\nStarting mock TEE server on {args.host}:{args.port}")
    print(f"Provider: {provider_type.value}")
    print(f"Model: {args.model_name} v{args.model_version}")
    print("\nEndpoints:")
    print(f"  Health:      http://{args.host}:{args.port}/health")
    print(f"  Attestation: http://{args.host}:{args.port}/attestation/create")
    print(f"  Inference:   http://{args.host}:{args.port}/inference")
    print(f"  Evidence:    http://{args.host}:{args.port}/evidence/bundle")
    print(f"  Statistics:  http://{args.host}:{args.port}/stats")
    print()
    
    app.run(host=args.host, port=args.port, debug=True)


if __name__ == '__main__':
    run_mock_server()