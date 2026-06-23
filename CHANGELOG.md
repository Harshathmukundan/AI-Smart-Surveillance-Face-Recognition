# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [1.0.0] - 2026-06-23

### Added
- Real-time face detection and recognition using OpenCV and face_recognition library
- GUI interface (Tkinter) for face registration with image upload
- SQLite database for storing user records with timestamps and locations
- Multi-source video input support (live camera & video files)
- Color-coded face bounding boxes (green = known, red = unknown)
- Tracking module for location-based monitoring
- GitHub Actions CI pipeline with Python linting and security scanning
- Comprehensive README with shields.io badges
- Contributing guidelines and security policy
- Requirements file for dependency management

### Security
- Added SECURITY.md with vulnerability reporting guidelines
- Integrated Bandit security scanning in CI pipeline
