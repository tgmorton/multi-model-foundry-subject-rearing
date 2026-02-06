#!/usr/bin/env python3
"""
NRP Documentation Scraper
Scrapes all User Guide documentation from nrp.ai and saves as markdown files.
"""

import os
import re
import time
import requests
from bs4 import BeautifulSoup
from markdownify import markdownify as md
from urllib.parse import urljoin

BASE_URL = "https://nrp.ai"
OUTPUT_DIR = "/Users/thomasmorton/Allie/nrp-docs"

# User Guide documentation URLs (from the documentation index)
USER_GUIDE_PAGES = [
    # Start section
    "/documentation/userdocs/start/getting-started",
    "/documentation/userdocs/start/using-nautilus",
    "/documentation/userdocs/start/hierarchy",
    "/documentation/userdocs/start/policies",
    "/documentation/userdocs/start/resources",
    "/documentation/userdocs/start/glossary",
    "/documentation/userdocs/start/faq",
    "/documentation/userdocs/start/support",

    # Tutorials section
    "/documentation/userdocs/tutorial/introduction",
    "/documentation/userdocs/tutorial/docker",
    "/documentation/userdocs/tutorial/basic",
    "/documentation/userdocs/tutorial/basic2",
    "/documentation/userdocs/tutorial/scheduling",
    "/documentation/userdocs/tutorial/jobs",
    "/documentation/userdocs/tutorial/images",
    "/documentation/userdocs/tutorial/storage",
    "/documentation/userdocs/tutorial/debugging",
    "/documentation/userdocs/tutorial/nrp-software",

    # Running - Beginner
    "/documentation/userdocs/running/gpu-pods",
    "/documentation/userdocs/running/long-idle",
    "/documentation/userdocs/running/monitoring",
    "/documentation/userdocs/running/jobs",
    "/documentation/userdocs/running/cpu-only",

    # Running - Intermediate
    "/documentation/userdocs/running/scheduling",
    "/documentation/userdocs/running/scripts",
    "/documentation/userdocs/running/ingress",
    "/documentation/userdocs/running/gateway",
    "/documentation/userdocs/running/special",
    "/documentation/userdocs/running/fast-img-download",
    "/documentation/userdocs/running/globus-connect",
    "/documentation/userdocs/running/kubernetes",
    "/documentation/userdocs/running/federation",
    "/documentation/userdocs/running/gui-desktop",
    "/documentation/userdocs/running/sci-img",
    "/documentation/userdocs/running/postgres",

    # Virtualization
    "/documentation/userdocs/running/virtualization-general",
    "/documentation/userdocs/running/virtualization-ubuntu",
    "/documentation/userdocs/running/virtualization-windows",

    # Distributed Computing
    "/documentation/userdocs/running/ray-cluster",
    "/documentation/userdocs/running/dask-cluster",
    "/documentation/userdocs/running/kubeflow",

    # Performance
    "/documentation/userdocs/running/io-jobs",
    "/documentation/userdocs/running/cpu-throttling",

    # Jupyter
    "/documentation/userdocs/jupyter/jupyterhub-service",
    "/documentation/userdocs/jupyter/jupyter-pod",
    "/documentation/userdocs/jupyter/jupyterhub",

    # Coder
    "/documentation/userdocs/coder/coder",
    "/documentation/userdocs/coder/deploy",

    # AI/LLMs
    "/documentation/userdocs/ai/llm-managed",
    "/documentation/userdocs/ai/llm-jupyterhub",
    "/documentation/userdocs/ai/qaic",
    "/documentation/userdocs/ai/vector-database",

    # Networks
    "/documentation/userdocs/networks/fabric",

    # FPGA
    "/documentation/userdocs/fpgas/vivado-vitis",
    "/documentation/userdocs/fpgas/esnet",
    "/documentation/userdocs/fpgas/esnet_development",
    "/documentation/userdocs/fpgas/esnet_building",
    "/documentation/userdocs/fpgas/esnet_running",

    # Storage
    "/documentation/userdocs/storage/intro",
    "/documentation/userdocs/storage/ceph",
    "/documentation/userdocs/storage/ceph-s3",
    "/documentation/userdocs/storage/cvmfs",
    "/documentation/userdocs/storage/local",
    "/documentation/userdocs/storage/linstor",
    "/documentation/userdocs/storage/nextcloud",
    "/documentation/userdocs/storage/syncthing",
    "/documentation/userdocs/storage/fuse",
    "/documentation/userdocs/storage/move-data",
    "/documentation/userdocs/storage/purging",
    "/documentation/userdocs/storage/jwt-credential",

    # Development
    "/documentation/userdocs/development/gitlab",
    "/documentation/userdocs/development/private-repos",
    "/documentation/userdocs/development/k8s-integration",
]


def get_page_content(url):
    """Fetch page and extract main documentation content."""
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')

        # Try to find the main content area (adjust selectors as needed)
        # Common patterns: main, article, .content, .documentation, etc.
        content = None
        for selector in ['main', 'article', '.content', '.documentation', '.docs-content', '#content']:
            content = soup.select_one(selector)
            if content:
                break

        if not content:
            # Fallback: get the body but remove nav, header, footer
            content = soup.find('body')
            if content:
                for tag in content.find_all(['nav', 'header', 'footer', 'aside', 'script', 'style']):
                    tag.decompose()

        return content, soup.title.string if soup.title else None
    except Exception as e:
        print(f"Error fetching {url}: {e}")
        return None, None


def html_to_markdown(html_content, page_url):
    """Convert HTML content to clean markdown."""
    if not html_content:
        return ""

    # Convert to markdown
    markdown = md(str(html_content),
                  heading_style="ATX",
                  code_language_callback=lambda el: el.get('class', [''])[0].replace('language-', '') if el.get('class') else '')

    # Clean up excessive whitespace
    markdown = re.sub(r'\n{3,}', '\n\n', markdown)
    markdown = markdown.strip()

    # Add source URL at the top
    markdown = f"**Source:** {page_url}\n\n{markdown}"

    return markdown


def get_output_path(url_path):
    """Generate output file path from URL path."""
    # Remove /documentation/userdocs/ prefix
    clean_path = url_path.replace('/documentation/userdocs/', '')

    # Split into directory and filename
    parts = clean_path.strip('/').split('/')

    if len(parts) >= 2:
        directory = parts[0]
        filename = parts[-1] + '.md'
    else:
        directory = ''
        filename = parts[0] + '.md'

    return os.path.join(OUTPUT_DIR, directory, filename)


def ensure_directory(filepath):
    """Ensure the directory for a file exists."""
    directory = os.path.dirname(filepath)
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)


def scrape_all_docs():
    """Main function to scrape all documentation pages."""
    print(f"Starting documentation scrape...")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Total pages to scrape: {len(USER_GUIDE_PAGES)}")
    print("-" * 50)

    successful = 0
    failed = 0

    for i, url_path in enumerate(USER_GUIDE_PAGES, 1):
        full_url = BASE_URL + url_path
        output_path = get_output_path(url_path)

        print(f"[{i}/{len(USER_GUIDE_PAGES)}] Scraping: {url_path}")

        # Fetch and convert
        content, title = get_page_content(full_url)

        if content:
            markdown = html_to_markdown(content, full_url)

            # Add title if found
            if title:
                clean_title = title.replace(' | NRP Nautilus', '').replace(' - NRP', '').strip()
                markdown = f"# {clean_title}\n\n{markdown}"

            # Save to file
            ensure_directory(output_path)
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(markdown)

            print(f"    -> Saved to: {output_path}")
            successful += 1
        else:
            print(f"    -> FAILED: Could not fetch content")
            failed += 1

        # Be nice to the server
        time.sleep(0.5)

    print("-" * 50)
    print(f"Scraping complete!")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")

    # Generate index file
    generate_index()


def generate_index():
    """Generate an index.md file listing all documentation."""
    index_content = """# NRP Nautilus Documentation Index

This folder contains scraped documentation from https://nrp.ai/documentation/

## User Guide

### Start
- [Getting Started](start/getting-started.md)
- [Using Nautilus](start/using-nautilus.md)
- [Hierarchy](start/hierarchy.md)
- [Policies](start/policies.md)
- [Resources](start/resources.md)
- [Glossary](start/glossary.md)
- [FAQ](start/faq.md)
- [Support](start/support.md)

### Tutorials
- [Introduction](tutorial/introduction.md)
- [Docker](tutorial/docker.md)
- [Basic](tutorial/basic.md)
- [Basic 2](tutorial/basic2.md)
- [Scheduling](tutorial/scheduling.md)
- [Jobs](tutorial/jobs.md)
- [Images](tutorial/images.md)
- [Storage](tutorial/storage.md)
- [Debugging](tutorial/debugging.md)
- [NRP Software](tutorial/nrp-software.md)

### Running (Beginner)
- [GPU Pods](running/gpu-pods.md)
- [Long Idle](running/long-idle.md)
- [Monitoring](running/monitoring.md)
- [Jobs](running/jobs.md)
- [CPU Only](running/cpu-only.md)

### Running (Intermediate)
- [Scheduling](running/scheduling.md)
- [Scripts](running/scripts.md)
- [Ingress](running/ingress.md)
- [Gateway](running/gateway.md)
- [Special](running/special.md)
- [Fast Image Download](running/fast-img-download.md)
- [Globus Connect](running/globus-connect.md)
- [Kubernetes](running/kubernetes.md)
- [Federation](running/federation.md)
- [GUI Desktop](running/gui-desktop.md)
- [Scientific Images](running/sci-img.md)
- [PostgreSQL](running/postgres.md)

### Virtualization
- [General](running/virtualization-general.md)
- [Ubuntu](running/virtualization-ubuntu.md)
- [Windows](running/virtualization-windows.md)

### Distributed Computing
- [Ray Cluster](running/ray-cluster.md)
- [Dask Cluster](running/dask-cluster.md)
- [Kubeflow](running/kubeflow.md)

### Performance
- [I/O Jobs](running/io-jobs.md)
- [CPU Throttling](running/cpu-throttling.md)

### Jupyter
- [JupyterHub Service](jupyter/jupyterhub-service.md)
- [Jupyter Pod](jupyter/jupyter-pod.md)
- [JupyterHub](jupyter/jupyterhub.md)

### Coder
- [Coder](coder/coder.md)
- [Deploy](coder/deploy.md)

### AI/LLMs
- [LLM Managed](ai/llm-managed.md)
- [LLM JupyterHub](ai/llm-jupyterhub.md)
- [QAIC](ai/qaic.md)
- [Vector Database](ai/vector-database.md)

### Networks
- [Fabric](networks/fabric.md)

### FPGA
- [Vivado Vitis](fpgas/vivado-vitis.md)
- [ESnet](fpgas/esnet.md)
- [ESnet Development](fpgas/esnet_development.md)
- [ESnet Building](fpgas/esnet_building.md)
- [ESnet Running](fpgas/esnet_running.md)

### Storage
- [Introduction](storage/intro.md)
- [Ceph](storage/ceph.md)
- [Ceph S3](storage/ceph-s3.md)
- [CVMFS](storage/cvmfs.md)
- [Local](storage/local.md)
- [Linstor](storage/linstor.md)
- [Nextcloud](storage/nextcloud.md)
- [Syncthing](storage/syncthing.md)
- [FUSE](storage/fuse.md)
- [Move Data](storage/move-data.md)
- [Purging](storage/purging.md)
- [JWT Credential](storage/jwt-credential.md)

### Development
- [GitLab](development/gitlab.md)
- [Private Repos](development/private-repos.md)
- [K8s Integration](development/k8s-integration.md)
"""

    index_path = os.path.join(OUTPUT_DIR, "index.md")
    with open(index_path, 'w', encoding='utf-8') as f:
        f.write(index_content)
    print(f"Generated index file: {index_path}")


if __name__ == "__main__":
    scrape_all_docs()
