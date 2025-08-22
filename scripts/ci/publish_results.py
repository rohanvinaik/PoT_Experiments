#!/usr/bin/env python3
"""
CI Results Publication

Publishes CI results, benchmarks, and evidence to various destinations
including GitHub Pages, artifact repositories, and reporting systems.
"""

import argparse
import json
import os
import sys
import shutil
import tempfile
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import subprocess
import zipfile

# Add parent directory for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


class ResultPublisher:
    """Publishes CI results to various destinations"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.temp_dir = Path(tempfile.mkdtemp(prefix='results_publish_'))
        self.published_artifacts = []
        
    def publish_to_github_pages(self, results_dir: str, branch: str = 'gh-pages') -> bool:
        """Publish results to GitHub Pages"""
        try:
            print(f"Publishing to GitHub Pages (branch: {branch})")
            
            # Check if we're in a git repository
            result = subprocess.run(
                ['git', 'rev-parse', '--git-dir'],
                capture_output=True, cwd=Path.cwd()
            )
            if result.returncode != 0:
                print("❌ Not in a git repository")
                return False
            
            # Create/checkout gh-pages branch
            subprocess.run(['git', 'fetch', 'origin', branch], capture_output=True)
            
            # Check if branch exists
            result = subprocess.run(
                ['git', 'rev-parse', '--verify', f'origin/{branch}'],
                capture_output=True
            )
            
            if result.returncode == 0:
                # Branch exists, checkout
                subprocess.run(['git', 'checkout', branch], check=True)
                subprocess.run(['git', 'pull', 'origin', branch], check=True)
            else:
                # Create new orphan branch
                subprocess.run(['git', 'checkout', '--orphan', branch], check=True)
                subprocess.run(['git', 'rm', '-rf', '.'], capture_output=True)
            
            # Copy results to gh-pages
            results_path = Path(results_dir)
            if not results_path.exists():
                print(f"❌ Results directory not found: {results_dir}")
                return False
            
            # Create directory structure
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            target_dir = Path('reports') / timestamp
            target_dir.mkdir(parents=True, exist_ok=True)
            
            # Copy files
            for item in results_path.iterdir():
                if item.is_file():
                    shutil.copy2(item, target_dir)
                elif item.is_dir():
                    shutil.copytree(item, target_dir / item.name, dirs_exist_ok=True)
            
            # Update index.html if it doesn't exist
            index_file = Path('index.html')
            if not index_file.exists():
                self._create_index_html(index_file)
            
            # Update latest symlink
            latest_link = Path('reports/latest')
            if latest_link.exists() or latest_link.is_symlink():
                latest_link.unlink()
            latest_link.symlink_to(timestamp)
            
            # Commit and push
            subprocess.run(['git', 'add', '.'], check=True)
            
            commit_message = f"CI Results: {timestamp}"
            subprocess.run(['git', 'commit', '-m', commit_message], check=True)
            subprocess.run(['git', 'push', 'origin', branch], check=True)
            
            # Return to original branch
            subprocess.run(['git', 'checkout', 'main'], capture_output=True)
            
            print(f"✅ Published to GitHub Pages: {target_dir}")
            return True
            
        except subprocess.CalledProcessError as e:
            print(f"❌ Git operation failed: {e}")
            return False
        except Exception as e:
            print(f"❌ Error publishing to GitHub Pages: {e}")
            return False
    
    def publish_to_artifact_store(self, artifacts_dir: str, store_config: Dict[str, Any]) -> bool:
        """Publish artifacts to artifact store"""
        try:
            store_type = store_config.get('type', 'local')
            
            if store_type == 'local':
                return self._publish_to_local_store(artifacts_dir, store_config)
            elif store_type == 'aws_s3':
                return self._publish_to_s3(artifacts_dir, store_config)
            elif store_type == 'gcs':
                return self._publish_to_gcs(artifacts_dir, store_config)
            else:
                print(f"❌ Unknown artifact store type: {store_type}")
                return False
                
        except Exception as e:
            print(f"❌ Error publishing to artifact store: {e}")
            return False
    
    def _publish_to_local_store(self, artifacts_dir: str, config: Dict[str, Any]) -> bool:
        """Publish to local artifact store"""
        store_path = Path(config.get('path', 'artifacts'))
        store_path.mkdir(parents=True, exist_ok=True)
        
        # Create timestamped directory
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        target_dir = store_path / timestamp
        
        # Copy artifacts
        artifacts_path = Path(artifacts_dir)
        if artifacts_path.exists():
            shutil.copytree(artifacts_path, target_dir)
            print(f"✅ Artifacts published to: {target_dir}")
            
            # Create latest symlink
            latest_link = store_path / 'latest'
            if latest_link.exists() or latest_link.is_symlink():
                latest_link.unlink()
            latest_link.symlink_to(timestamp)
            
            return True
        else:
            print(f"❌ Artifacts directory not found: {artifacts_dir}")
            return False
    
    def _publish_to_s3(self, artifacts_dir: str, config: Dict[str, Any]) -> bool:
        """Publish to AWS S3"""
        try:
            import boto3
            from botocore.exceptions import NoCredentialsError
        except ImportError:
            print("❌ boto3 not available for S3 publishing")
            return False
        
        try:
            bucket = config.get('bucket')
            prefix = config.get('prefix', 'ci-artifacts')
            
            if not bucket:
                print("❌ S3 bucket not specified")
                return False
            
            s3 = boto3.client('s3')
            
            # Upload files
            artifacts_path = Path(artifacts_dir)
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            
            for file_path in artifacts_path.rglob('*'):
                if file_path.is_file():
                    relative_path = file_path.relative_to(artifacts_path)
                    s3_key = f"{prefix}/{timestamp}/{relative_path}"
                    
                    s3.upload_file(str(file_path), bucket, s3_key)
                    print(f"📤 Uploaded: s3://{bucket}/{s3_key}")
            
            print(f"✅ Artifacts published to S3: s3://{bucket}/{prefix}/{timestamp}")
            return True
            
        except NoCredentialsError:
            print("❌ AWS credentials not found")
            return False
        except Exception as e:
            print(f"❌ Error uploading to S3: {e}")
            return False
    
    def _publish_to_gcs(self, artifacts_dir: str, config: Dict[str, Any]) -> bool:
        """Publish to Google Cloud Storage"""
        try:
            from google.cloud import storage
        except ImportError:
            print("❌ google-cloud-storage not available for GCS publishing")
            return False
        
        try:
            bucket_name = config.get('bucket')
            prefix = config.get('prefix', 'ci-artifacts')
            
            if not bucket_name:
                print("❌ GCS bucket not specified")
                return False
            
            client = storage.Client()
            bucket = client.bucket(bucket_name)
            
            # Upload files
            artifacts_path = Path(artifacts_dir)
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            
            for file_path in artifacts_path.rglob('*'):
                if file_path.is_file():
                    relative_path = file_path.relative_to(artifacts_path)
                    blob_name = f"{prefix}/{timestamp}/{relative_path}"
                    
                    blob = bucket.blob(blob_name)
                    blob.upload_from_filename(str(file_path))
                    print(f"📤 Uploaded: gs://{bucket_name}/{blob_name}")
            
            print(f"✅ Artifacts published to GCS: gs://{bucket_name}/{prefix}/{timestamp}")
            return True
            
        except Exception as e:
            print(f"❌ Error uploading to GCS: {e}")
            return False
    
    def publish_notifications(self, results: Dict[str, Any], notification_config: Dict[str, Any]) -> bool:
        """Send notifications about CI results"""
        try:
            notification_type = notification_config.get('type')
            
            if notification_type == 'slack':
                return self._send_slack_notification(results, notification_config)
            elif notification_type == 'email':
                return self._send_email_notification(results, notification_config)
            elif notification_type == 'webhook':
                return self._send_webhook_notification(results, notification_config)
            else:
                print(f"❌ Unknown notification type: {notification_type}")
                return False
                
        except Exception as e:
            print(f"❌ Error sending notifications: {e}")
            return False
    
    def _send_slack_notification(self, results: Dict[str, Any], config: Dict[str, Any]) -> bool:
        """Send Slack notification"""
        try:
            import requests
        except ImportError:
            print("❌ requests not available for Slack notifications")
            return False
        
        webhook_url = config.get('webhook_url')
        if not webhook_url:
            print("❌ Slack webhook URL not provided")
            return False
        
        # Create message
        status = "✅ Success" if results.get('success', True) else "❌ Failed"
        commit = results.get('commit_sha', 'unknown')[:8]
        
        message = {
            "text": f"CI Pipeline {status}",
            "blocks": [
                {
                    "type": "header",
                    "text": {
                        "type": "plain_text",
                        "text": f"CI Pipeline {status}"
                    }
                },
                {
                    "type": "section",
                    "fields": [
                        {
                            "type": "mrkdwn",
                            "text": f"*Commit:* {commit}"
                        },
                        {
                            "type": "mrkdwn",
                            "text": f"*Branch:* {results.get('branch', 'unknown')}"
                        }
                    ]
                }
            ]
        }
        
        # Add test results if available
        if 'test_results' in results:
            test_results = results['test_results']
            message["blocks"].append({
                "type": "section",
                "fields": [
                    {
                        "type": "mrkdwn",
                        "text": f"*Tests:* {test_results.get('passed', 0)} passed, {test_results.get('failed', 0)} failed"
                    }
                ]
            })
        
        response = requests.post(webhook_url, json=message)
        if response.status_code == 200:
            print("✅ Slack notification sent")
            return True
        else:
            print(f"❌ Slack notification failed: {response.status_code}")
            return False
    
    def _send_email_notification(self, results: Dict[str, Any], config: Dict[str, Any]) -> bool:
        """Send email notification"""
        try:
            import smtplib
            from email.mime.text import MimeText
            from email.mime.multipart import MimeMultipart
        except ImportError:
            print("❌ Email libraries not available")
            return False
        
        smtp_server = config.get('smtp_server')
        smtp_port = config.get('smtp_port', 587)
        username = config.get('username')
        password = config.get('password')
        recipients = config.get('recipients', [])
        
        if not all([smtp_server, username, password, recipients]):
            print("❌ Email configuration incomplete")
            return False
        
        try:
            # Create message
            msg = MimeMultipart()
            msg['From'] = username
            msg['To'] = ', '.join(recipients)
            
            status = "Success" if results.get('success', True) else "Failed"
            commit = results.get('commit_sha', 'unknown')[:8]
            msg['Subject'] = f"CI Pipeline {status} - {commit}"
            
            # Create body
            body = f"""
CI Pipeline Status: {status}
Commit: {results.get('commit_sha', 'unknown')}
Branch: {results.get('branch', 'unknown')}
Timestamp: {results.get('timestamp', datetime.utcnow().isoformat())}

{json.dumps(results, indent=2)}
            """
            
            msg.attach(MimeText(body, 'plain'))
            
            # Send email
            server = smtplib.SMTP(smtp_server, smtp_port)
            server.starttls()
            server.login(username, password)
            text = msg.as_string()
            server.sendmail(username, recipients, text)
            server.quit()
            
            print("✅ Email notification sent")
            return True
            
        except Exception as e:
            print(f"❌ Error sending email: {e}")
            return False
    
    def _send_webhook_notification(self, results: Dict[str, Any], config: Dict[str, Any]) -> bool:
        """Send webhook notification"""
        try:
            import requests
        except ImportError:
            print("❌ requests not available for webhook notifications")
            return False
        
        url = config.get('url')
        if not url:
            print("❌ Webhook URL not provided")
            return False
        
        headers = config.get('headers', {})
        headers.setdefault('Content-Type', 'application/json')
        
        payload = {
            'timestamp': datetime.utcnow().isoformat(),
            'source': 'ci_pipeline',
            'results': results
        }
        
        try:
            response = requests.post(url, json=payload, headers=headers, timeout=30)
            if response.status_code < 400:
                print("✅ Webhook notification sent")
                return True
            else:
                print(f"❌ Webhook notification failed: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ Error sending webhook: {e}")
            return False
    
    def _create_index_html(self, index_file: Path):
        """Create basic index.html for GitHub Pages"""
        html_content = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>PoT CI Reports</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; }
        h1 { color: #333; }
        .report-list { list-style-type: none; padding: 0; }
        .report-item { margin: 10px 0; padding: 10px; border: 1px solid #ddd; border-radius: 5px; }
        .timestamp { color: #666; font-size: 0.9em; }
    </style>
</head>
<body>
    <h1>PoT Framework CI Reports</h1>
    <p>Automated CI reports and benchmarks for the Proof-of-Training framework.</p>
    
    <h2>Latest Reports</h2>
    <ul class="report-list">
        <li class="report-item">
            <a href="reports/latest/">Latest Report</a>
            <div class="timestamp">Most recent CI run</div>
        </li>
    </ul>
    
    <h2>All Reports</h2>
    <p>Browse the <a href="reports/">reports directory</a> for historical reports.</p>
    
    <p><em>Generated automatically by CI pipeline</em></p>
</body>
</html>"""
        
        with open(index_file, 'w') as f:
            f.write(html_content)
    
    def cleanup(self):
        """Clean up temporary files"""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)


def load_config(config_file: str) -> Dict[str, Any]:
    """Load configuration from file"""
    config_path = Path(config_file)
    if not config_path.exists():
        return {}
    
    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"⚠️  Error loading config: {e}")
        return {}


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Publish CI results and artifacts'
    )
    parser.add_argument(
        '--results-dir',
        type=str,
        required=True,
        help='Directory containing CI results'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='ci_config.json',
        help='Configuration file (default: ci_config.json)'
    )
    parser.add_argument(
        '--github-pages',
        action='store_true',
        help='Publish to GitHub Pages'
    )
    parser.add_argument(
        '--artifacts',
        action='store_true',
        help='Publish to artifact store'
    )
    parser.add_argument(
        '--notifications',
        action='store_true',
        help='Send notifications'
    )
    parser.add_argument(
        '--all',
        action='store_true',
        help='Publish to all configured destinations'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose output'
    )
    
    args = parser.parse_args()
    
    try:
        # Load configuration
        config = load_config(args.config)
        
        # Create publisher
        publisher = ResultPublisher(config)
        
        # Load results summary if available
        results_summary = {}
        summary_file = Path(args.results_dir) / 'summary.json'
        if summary_file.exists():
            with open(summary_file, 'r') as f:
                results_summary = json.load(f)
        
        success = True
        
        # Publish to GitHub Pages
        if args.github_pages or args.all:
            gh_pages_config = config.get('github_pages', {})
            if gh_pages_config.get('enabled', True):
                branch = gh_pages_config.get('branch', 'gh-pages')
                if not publisher.publish_to_github_pages(args.results_dir, branch):
                    success = False
        
        # Publish artifacts
        if args.artifacts or args.all:
            artifact_config = config.get('artifacts', {})
            if artifact_config.get('enabled', True):
                if not publisher.publish_to_artifact_store(args.results_dir, artifact_config):
                    success = False
        
        # Send notifications
        if args.notifications or args.all:
            notification_configs = config.get('notifications', [])
            for notification_config in notification_configs:
                if notification_config.get('enabled', True):
                    if not publisher.publish_notifications(results_summary, notification_config):
                        success = False
        
        # Cleanup
        publisher.cleanup()
        
        if success:
            print("\n✅ All publishing operations completed successfully")
            sys.exit(0)
        else:
            print("\n❌ Some publishing operations failed")
            sys.exit(1)
        
    except Exception as e:
        print(f"❌ Error publishing results: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()