#!/usr/bin/env python3
"""
ZK Binary Version Information System

Manages version information, build metadata, and compatibility tracking
for ZK proof system binaries and components.
"""

import os
import sys
import json
import subprocess
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict


@dataclass
class BinaryVersion:
    """Version information for a ZK binary"""
    binary_name: str
    version: str
    git_commit: Optional[str]
    build_timestamp: str
    rust_version: str
    cargo_version: str
    profile: str  # debug or release
    target_triple: str
    file_hash: str
    file_size: int
    dependencies: Dict[str, str]


@dataclass 
class ZKSystemVersion:
    """Complete ZK system version information"""
    system_version: str
    python_version: str
    binaries: List[BinaryVersion]
    build_info: Dict[str, Any]
    compatibility: Dict[str, Any]
    timestamp: str


class VersionManager:
    """Manages version information for ZK system components"""
    
    def __init__(self):
        self.project_root = Path(__file__).parent.parent.parent
        self.prover_dir = self.project_root / "pot/zk/prover_halo2"
        self.version_file = self.project_root / "pot/zk/version_info.json"
    
    def get_git_commit_hash(self) -> Optional[str]:
        """Get current git commit hash"""
        try:
            result = subprocess.run(
                ['git', 'rev-parse', 'HEAD'],
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=10
            )
            if result.returncode == 0:
                return result.stdout.strip()
        except:
            pass
        return None
    
    def get_git_branch(self) -> Optional[str]:
        """Get current git branch"""
        try:
            result = subprocess.run(
                ['git', 'rev-parse', '--abbrev-ref', 'HEAD'],
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=10
            )
            if result.returncode == 0:
                return result.stdout.strip()
        except:
            pass
        return None
    
    def get_rust_version_info(self) -> Dict[str, str]:
        """Get detailed Rust version information"""
        info = {}
        
        try:
            # Get rustc version
            result = subprocess.run(
                ['rustc', '--version'],
                capture_output=True,
                text=True,
                timeout=10
            )
            if result.returncode == 0:
                info['rustc'] = result.stdout.strip()
        except:
            pass
        
        try:
            # Get cargo version
            result = subprocess.run(
                ['cargo', '--version'],
                capture_output=True,
                text=True,
                timeout=10
            )
            if result.returncode == 0:
                info['cargo'] = result.stdout.strip()
        except:
            pass
        
        try:
            # Get target triple
            result = subprocess.run(
                ['rustc', '--print', 'target-list'],
                capture_output=True,
                text=True,
                timeout=10
            )
            if result.returncode == 0:
                # Get default target
                default_result = subprocess.run(
                    ['rustc', '-vV'],
                    capture_output=True,
                    text=True,
                    timeout=10
                )
                if default_result.returncode == 0:
                    for line in default_result.stdout.split('\n'):
                        if line.startswith('host: '):
                            info['target_triple'] = line.split(': ')[1]
                            break
        except:
            pass
        
        return info
    
    def get_file_info(self, file_path: Path) -> Dict[str, Any]:
        """Get file hash and metadata"""
        if not file_path.exists():
            return {'exists': False}
        
        # Calculate file hash
        hasher = hashlib.sha256()
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hasher.update(chunk)
        
        stat = file_path.stat()
        
        return {
            'exists': True,
            'sha256': hasher.hexdigest(),
            'size_bytes': stat.st_size,
            'modified_timestamp': stat.st_mtime,
            'permissions': oct(stat.st_mode)[-3:]
        }
    
    def get_cargo_dependencies(self) -> Dict[str, str]:
        """Get Cargo.toml dependencies with versions"""
        cargo_toml = self.prover_dir / "Cargo.toml"
        dependencies = {}
        
        if not cargo_toml.exists():
            return dependencies
        
        try:
            # Use cargo metadata to get actual resolved versions
            result = subprocess.run(
                ['cargo', 'metadata', '--format-version', '1'],
                cwd=self.prover_dir,
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode == 0:
                metadata = json.loads(result.stdout)
                for package in metadata.get('packages', []):
                    dependencies[package['name']] = package['version']
        except:
            # Fallback: parse Cargo.toml manually
            try:
                with open(cargo_toml, 'r') as f:
                    content = f.read()
                
                in_dependencies = False
                for line in content.split('\n'):
                    line = line.strip()
                    if line == '[dependencies]':
                        in_dependencies = True
                        continue
                    elif line.startswith('[') and in_dependencies:
                        break
                    elif in_dependencies and '=' in line:
                        parts = line.split('=')
                        if len(parts) >= 2:
                            dep_name = parts[0].strip()
                            version = parts[1].strip().strip('"\'')
                            dependencies[dep_name] = version
            except:
                pass
        
        return dependencies
    
    def extract_binary_version(self, binary_path: Path) -> Optional[BinaryVersion]:
        """Extract version information from a binary"""
        if not binary_path.exists():
            return None
        
        # Get file info
        file_info = self.get_file_info(binary_path)
        if not file_info['exists']:
            return None
        
        # Try to get version from binary
        version = "unknown"
        try:
            result = subprocess.run(
                [str(binary_path), '--version'],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                version = result.stdout.strip()
        except:
            pass
        
        # Get Rust version info
        rust_info = self.get_rust_version_info()
        
        # Determine profile from path
        profile = "release" if "release" in str(binary_path) else "debug"
        
        # Get dependencies
        dependencies = self.get_cargo_dependencies()
        
        return BinaryVersion(
            binary_name=binary_path.name,
            version=version,
            git_commit=self.get_git_commit_hash(),
            build_timestamp=datetime.fromtimestamp(file_info['modified_timestamp'], timezone.utc).isoformat(),
            rust_version=rust_info.get('rustc', 'unknown'),
            cargo_version=rust_info.get('cargo', 'unknown'),
            profile=profile,
            target_triple=rust_info.get('target_triple', 'unknown'),
            file_hash=file_info['sha256'],
            file_size=file_info['size_bytes'],
            dependencies=dependencies
        )
    
    def scan_all_binaries(self) -> List[BinaryVersion]:
        """Scan all ZK binaries and extract version information"""
        binary_paths = [
            self.prover_dir / "target/debug/prove_sgd_stdin",
            self.prover_dir / "target/debug/verify_sgd_stdin", 
            self.prover_dir / "target/debug/prove_lora_stdin",
            self.prover_dir / "target/debug/verify_lora_stdin",
            self.prover_dir / "target/release/prove_sgd_stdin",
            self.prover_dir / "target/release/verify_sgd_stdin",
            self.prover_dir / "target/release/prove_lora_stdin",
            self.prover_dir / "target/release/verify_lora_stdin",
        ]
        
        versions = []
        for path in binary_paths:
            version_info = self.extract_binary_version(path)
            if version_info:
                versions.append(version_info)
        
        return versions
    
    def generate_build_info(self) -> Dict[str, Any]:
        """Generate comprehensive build information"""
        return {
            'git_commit': self.get_git_commit_hash(),
            'git_branch': self.get_git_branch(),
            'build_host': os.uname().nodename if hasattr(os, 'uname') else 'unknown',
            'build_user': os.getenv('USER', 'unknown'),
            'build_timestamp': datetime.now(timezone.utc).isoformat(),
            'rust_info': self.get_rust_version_info(),
            'python_version': sys.version,
            'platform': sys.platform,
            'architecture': os.uname().machine if hasattr(os, 'uname') else 'unknown'
        }
    
    def check_compatibility(self, binaries: List[BinaryVersion]) -> Dict[str, Any]:
        """Check compatibility between binaries and system"""
        compatibility = {
            'rust_version_consistent': True,
            'git_commit_consistent': True,
            'profile_mix': False,
            'missing_binaries': [],
            'version_mismatches': [],
            'recommendations': []
        }
        
        expected_binaries = [
            'prove_sgd_stdin', 'verify_sgd_stdin',
            'prove_lora_stdin', 'verify_lora_stdin'
        ]
        
        # Check for missing binaries
        found_binaries = set(b.binary_name for b in binaries)
        missing = set(expected_binaries) - found_binaries
        compatibility['missing_binaries'] = list(missing)
        
        if missing:
            compatibility['recommendations'].append(f"Build missing binaries: {', '.join(missing)}")
        
        # Check version consistency
        if binaries:
            first_rust = binaries[0].rust_version
            first_commit = binaries[0].git_commit
            
            profiles = set()
            for binary in binaries:
                profiles.add(binary.profile)
                
                if binary.rust_version != first_rust:
                    compatibility['rust_version_consistent'] = False
                    compatibility['version_mismatches'].append(
                        f"{binary.binary_name} has different Rust version"
                    )
                
                if binary.git_commit != first_commit:
                    compatibility['git_commit_consistent'] = False
                    compatibility['version_mismatches'].append(
                        f"{binary.binary_name} built from different git commit"
                    )
            
            compatibility['profile_mix'] = len(profiles) > 1
            if compatibility['profile_mix']:
                compatibility['recommendations'].append(
                    "Consider using consistent build profile (all debug or all release)"
                )
        
        return compatibility
    
    def generate_system_version(self) -> ZKSystemVersion:
        """Generate complete system version information"""
        binaries = self.scan_all_binaries()
        build_info = self.generate_build_info()
        compatibility = self.check_compatibility(binaries)
        
        return ZKSystemVersion(
            system_version="1.0.0",  # Could be read from a VERSION file
            python_version=sys.version.split()[0],
            binaries=binaries,
            build_info=build_info,
            compatibility=compatibility,
            timestamp=datetime.now(timezone.utc).isoformat()
        )
    
    def save_version_info(self, system_version: Optional[ZKSystemVersion] = None):
        """Save version information to JSON file"""
        if system_version is None:
            system_version = self.generate_system_version()
        
        self.version_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(self.version_file, 'w') as f:
            json.dump(asdict(system_version), f, indent=2, default=str)
    
    def load_version_info(self) -> Optional[ZKSystemVersion]:
        """Load version information from JSON file"""
        if not self.version_file.exists():
            return None
        
        try:
            with open(self.version_file, 'r') as f:
                data = json.load(f)
            
            # Convert back to dataclass (simplified)
            return ZKSystemVersion(**data)
        except:
            return None
    
    def embed_version_in_binaries(self):
        """Embed version information in Rust binaries during build"""
        # This would modify Cargo.toml to include build.rs script
        # that embeds git commit, build time, etc. as constants
        
        build_rs_content = '''
fn main() {
    // Embed git commit hash
    if let Some(git_hash) = git_commit_hash() {
        println!("cargo:rustc-env=GIT_COMMIT_HASH={}", git_hash);
    }
    
    // Embed build timestamp
    let build_time = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs();
    println!("cargo:rustc-env=BUILD_TIMESTAMP={}", build_time);
    
    // Embed Rust version
    println!("cargo:rustc-env=RUST_VERSION={}", env!("RUSTC_VERSION"));
}

fn git_commit_hash() -> Option<String> {
    std::process::Command::new("git")
        .args(&["rev-parse", "HEAD"])
        .output()
        .ok()
        .and_then(|output| {
            if output.status.success() {
                String::from_utf8(output.stdout).ok()
            } else {
                None
            }
        })
        .map(|s| s.trim().to_string())
}
'''
        
        build_rs_path = self.prover_dir / "build.rs"
        
        if not build_rs_path.exists():
            with open(build_rs_path, 'w') as f:
                f.write(build_rs_content)
            
            print(f"Created build.rs for version embedding at {build_rs_path}")
    
    def generate_version_flags(self) -> str:
        """Generate --version flag implementation for Rust binaries"""
        return '''
// Add this to your main.rs files

use std::env;

const VERSION: &str = env!("CARGO_PKG_VERSION");
const GIT_COMMIT: &str = env!("GIT_COMMIT_HASH");
const BUILD_TIME: &str = env!("BUILD_TIMESTAMP");
const RUST_VERSION: &str = env!("RUST_VERSION");

fn print_version() {
    println!("{} version {}", env!("CARGO_PKG_NAME"), VERSION);
    println!("Git commit: {}", GIT_COMMIT);
    println!("Build time: {}", BUILD_TIME);
    println!("Rust version: {}", RUST_VERSION);
}

// In your main function, add:
// if args.contains(&"--version".to_string()) {
//     print_version();
//     return;
// }
'''


def get_system_version() -> ZKSystemVersion:
    """Get current ZK system version information"""
    manager = VersionManager()
    return manager.generate_system_version()


def save_version_snapshot():
    """Save current version information snapshot"""
    manager = VersionManager()
    manager.save_version_info()
    print(f"Version information saved to {manager.version_file}")


def main():
    """Command-line interface for version management"""
    import argparse
    
    parser = argparse.ArgumentParser(description='ZK System Version Management')
    parser.add_argument('command', choices=['info', 'save', 'embed', 'generate-flags'],
                       help='Command to execute')
    parser.add_argument('--format', choices=['json', 'human'], default='human',
                       help='Output format for info command')
    parser.add_argument('--output', '-o', help='Output file for info command')
    
    args = parser.parse_args()
    
    manager = VersionManager()
    
    if args.command == 'info':
        system_version = manager.generate_system_version()
        
        if args.format == 'json':
            output = json.dumps(asdict(system_version), indent=2, default=str)
        else:
            # Human readable format
            output = f"""
ZK System Version Information
=============================

System Version: {system_version.system_version}
Python Version: {system_version.python_version}
Timestamp: {system_version.timestamp}

Build Information:
- Git Commit: {system_version.build_info.get('git_commit', 'unknown')}
- Git Branch: {system_version.build_info.get('git_branch', 'unknown')}
- Build Host: {system_version.build_info.get('build_host', 'unknown')}
- Build User: {system_version.build_info.get('build_user', 'unknown')}

Binaries Found: {len(system_version.binaries)}
"""
            for binary in system_version.binaries:
                output += f"- {binary.binary_name} ({binary.profile}): {binary.version}\n"
                output += f"  Hash: {binary.file_hash[:16]}...\n"
                output += f"  Size: {binary.file_size} bytes\n"
            
            output += f"\nCompatibility:\n"
            for key, value in system_version.compatibility.items():
                if key != 'recommendations':
                    output += f"- {key}: {value}\n"
            
            if system_version.compatibility['recommendations']:
                output += "\nRecommendations:\n"
                for rec in system_version.compatibility['recommendations']:
                    output += f"• {rec}\n"
        
        if args.output:
            with open(args.output, 'w') as f:
                if args.format == 'json':
                    json.dump(asdict(system_version), f, indent=2, default=str)
                else:
                    f.write(output)
            print(f"Version info saved to {args.output}")
        else:
            print(output)
    
    elif args.command == 'save':
        manager.save_version_info()
        print(f"Version information saved to {manager.version_file}")
    
    elif args.command == 'embed':
        manager.embed_version_in_binaries()
        print("Version embedding setup complete")
        print("Rebuild binaries with: cd pot/zk/prover_halo2 && cargo build --release")
    
    elif args.command == 'generate-flags':
        print(manager.generate_version_flags())


if __name__ == "__main__":
    main()