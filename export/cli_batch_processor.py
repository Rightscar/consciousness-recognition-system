"""
CLI Batch Processor
==================

Provides command-line interface and headless batch processing capabilities
for processing large books and documents unattended.

Features:
- Complete CLI interface with argument parsing
- Batch processing of multiple files
- Progress tracking and logging
- Configurable processing parameters
- Resume capability for interrupted jobs
- Parallel processing support
- Output format customization
"""

import argparse
import sys
import os
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict
import signal
from datetime import datetime

# Import core modules
from modules.logger import get_logger
from ingestion.advanced_pdf_processor import AdvancedPDFProcessor
from extraction.intelligent_text_processor import IntelligentTextProcessor
from enhancement.resilient_enhancer import ResilientEnhancer
from quality.quality_scoring_dashboard import QualityScoringDashboard
from export.export_utils import ExportUtilities

@dataclass
class BatchConfig:
    """Configuration for batch processing."""
    input_path: str
    output_path: str
    format: str = 'jsonl'
    tone: str = 'universal_wisdom'
    quality_threshold: float = 0.7
    chunk_size: int = 1000
    max_workers: int = 4
    resume: bool = False
    overwrite: bool = False
    verbose: bool = False
    dry_run: bool = False
    include_metadata: bool = True
    export_format: List[str] = None
    
    def __post_init__(self):
        if self.export_format is None:
            self.export_format = ['jsonl']

@dataclass
class ProcessingResult:
    """Result of processing a single file."""
    file_path: str
    success: bool
    output_path: str = None
    error: str = None
    processing_time: float = 0.0
    chunks_processed: int = 0
    quality_score: float = 0.0
    metadata: Dict[str, Any] = None

class CLIBatchProcessor:
    """
    Command-line batch processor for unattended document processing.
    
    Supports processing single files or entire directories with configurable
    parameters, progress tracking, and resume capabilities.
    """
    
    def __init__(self, config: BatchConfig):
        """
        Initialize the CLI batch processor.
        
        Args:
            config: Batch processing configuration
        """
        self.config = config
        self.logger = get_logger("cli_batch_processor")
        
        # Initialize processing components
        self.pdf_processor = AdvancedPDFProcessor()
        self.text_processor = IntelligentTextProcessor()
        self.enhancer = ResilientEnhancer()
        self.quality_dashboard = QualityScoringDashboard()
        self.export_utils = ExportUtilities()
        
        # Processing state
        self.processed_files = set()
        self.failed_files = set()
        self.results = []
        self.start_time = None
        self.interrupted = False
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        self.logger.info("CLI batch processor initialized")
    
    def process_batch(self) -> Dict[str, Any]:
        """
        Process files in batch mode.
        
        Returns:
            Dictionary with batch processing results and statistics
        """
        try:
            self.start_time = time.time()
            self.logger.info(f"Starting batch processing: {self.config.input_path}")
            
            # Validate configuration
            self._validate_config()
            
            # Setup output directory
            self._setup_output_directory()
            
            # Load resume state if requested
            if self.config.resume:
                self._load_resume_state()
            
            # Get list of files to process
            files_to_process = self._get_files_to_process()
            
            if not files_to_process:
                raise ValueError("No files found to process")
            
            self.logger.info(f"Found {len(files_to_process)} files to process")
            
            if self.config.dry_run:
                return self._dry_run_report(files_to_process)
            
            # Process files
            if self.config.max_workers > 1:
                results = self._process_files_parallel(files_to_process)
            else:
                results = self._process_files_sequential(files_to_process)
            
            # Generate final report
            final_report = self._generate_final_report(results)
            
            # Save processing state
            self._save_processing_state(final_report)
            
            self.logger.info("Batch processing completed successfully")
            return final_report
            
        except KeyboardInterrupt:
            self.logger.warning("Batch processing interrupted by user")
            self.interrupted = True
            return self._generate_interruption_report()
        except Exception as e:
            self.logger.error(f"Batch processing failed: {str(e)}")
            raise
    
    def _validate_config(self):
        """Validate the batch processing configuration."""
        # Check input path exists
        input_path = Path(self.config.input_path)
        if not input_path.exists():
            raise FileNotFoundError(f"Input path does not exist: {input_path}")
        
        # Validate output format
        valid_formats = ['jsonl', 'json', 'csv', 'xlsx', 'txt']
        if self.config.format not in valid_formats:
            raise ValueError(f"Invalid format: {self.config.format}. Valid formats: {valid_formats}")
        
        # Validate quality threshold
        if not 0.0 <= self.config.quality_threshold <= 1.0:
            raise ValueError("Quality threshold must be between 0.0 and 1.0")
        
        # Validate chunk size
        if self.config.chunk_size < 100:
            raise ValueError("Chunk size must be at least 100")
        
        # Validate max workers
        if self.config.max_workers < 1:
            raise ValueError("Max workers must be at least 1")
    
    def _setup_output_directory(self):
        """Setup the output directory structure."""
        output_path = Path(self.config.output_path)
        
        # Create output directory
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        (output_path / 'data').mkdir(exist_ok=True)
        (output_path / 'metadata').mkdir(exist_ok=True)
        (output_path / 'logs').mkdir(exist_ok=True)
        (output_path / 'state').mkdir(exist_ok=True)
        
        self.logger.info(f"Output directory setup: {output_path}")
    
    def _get_files_to_process(self) -> List[Path]:
        """Get list of files to process."""
        input_path = Path(self.config.input_path)
        files = []
        
        if input_path.is_file():
            # Single file
            files = [input_path]
        elif input_path.is_dir():
            # Directory - find all supported files
            supported_extensions = ['.pdf', '.txt', '.docx', '.epub']
            for ext in supported_extensions:
                files.extend(input_path.rglob(f'*{ext}'))
        
        # Filter out already processed files if resuming
        if self.config.resume:
            files = [f for f in files if str(f) not in self.processed_files]
        
        # Sort for consistent processing order
        return sorted(files)
    
    def _process_files_sequential(self, files: List[Path]) -> List[ProcessingResult]:
        """Process files sequentially."""
        results = []
        
        for i, file_path in enumerate(files):
            if self.interrupted:
                break
            
            self.logger.info(f"Processing file {i+1}/{len(files)}: {file_path.name}")
            
            result = self._process_single_file(file_path)
            results.append(result)
            
            # Save intermediate state
            if i % 10 == 0:  # Save every 10 files
                self._save_intermediate_state(results)
        
        return results
    
    def _process_files_parallel(self, files: List[Path]) -> List[ProcessingResult]:
        """Process files in parallel."""
        results = []
        
        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            # Submit all jobs
            future_to_file = {
                executor.submit(self._process_single_file, file_path): file_path
                for file_path in files
            }
            
            # Process completed jobs
            for i, future in enumerate(as_completed(future_to_file)):
                if self.interrupted:
                    # Cancel remaining futures
                    for f in future_to_file:
                        f.cancel()
                    break
                
                file_path = future_to_file[future]
                
                try:
                    result = future.result()
                    results.append(result)
                    
                    self.logger.info(f"Completed {i+1}/{len(files)}: {file_path.name}")
                    
                    # Save intermediate state
                    if i % 10 == 0:
                        self._save_intermediate_state(results)
                        
                except Exception as e:
                    self.logger.error(f"Failed to process {file_path}: {str(e)}")
                    results.append(ProcessingResult(
                        file_path=str(file_path),
                        success=False,
                        error=str(e)
                    ))
        
        return results
    
    def _process_single_file(self, file_path: Path) -> ProcessingResult:
        """Process a single file."""
        start_time = time.time()
        
        try:
            self.logger.debug(f"Starting processing: {file_path}")
            
            # Step 1: Extract text from file
            if file_path.suffix.lower() == '.pdf':
                extraction_result = self.pdf_processor.extract_with_ocr_fallback(str(file_path))
                raw_text = extraction_result['text']
                extraction_metadata = extraction_result['metadata']
            else:
                # Handle other file types
                raw_text = self._extract_text_from_file(file_path)
                extraction_metadata = {'method': 'direct_read'}
            
            if not raw_text.strip():
                raise ValueError("No text extracted from file")
            
            # Step 2: Normalize text
            normalization_result = self.text_processor.normalize_multilingual_text(raw_text)
            normalized_text = normalization_result['normalized_text']
            
            # Step 3: Chunk text
            chunks = self._chunk_text(normalized_text)
            
            if not chunks:
                raise ValueError("No chunks created from text")
            
            # Step 4: Enhance chunks
            enhanced_chunks = []
            for chunk in chunks:
                enhanced_chunk = self.enhancer.enhance_content(
                    chunk, 
                    tone=self.config.tone,
                    quality_threshold=self.config.quality_threshold
                )
                enhanced_chunks.append(enhanced_chunk)
            
            # Step 5: Quality assessment
            quality_scores = []
            for original, enhanced in zip(chunks, enhanced_chunks):
                score = self.quality_dashboard.calculate_overall_quality_score(
                    original, enhanced
                )
                quality_scores.append(score)
            
            avg_quality = sum(quality_scores) / len(quality_scores)
            
            # Step 6: Export results
            output_file = self._generate_output_filename(file_path)
            export_data = self._prepare_export_data(
                chunks, enhanced_chunks, quality_scores, file_path
            )
            
            self.export_utils.export_to_format(
                export_data,
                str(output_file),
                self.config.format
            )
            
            # Step 7: Generate metadata
            metadata = self._generate_file_metadata(
                file_path, extraction_metadata, normalization_result,
                len(chunks), avg_quality
            )
            
            if self.config.include_metadata:
                metadata_file = output_file.with_suffix('.metadata.json')
                with open(metadata_file, 'w', encoding='utf-8') as f:
                    json.dump(metadata, f, indent=2, ensure_ascii=False)
            
            processing_time = time.time() - start_time
            
            result = ProcessingResult(
                file_path=str(file_path),
                success=True,
                output_path=str(output_file),
                processing_time=processing_time,
                chunks_processed=len(chunks),
                quality_score=avg_quality,
                metadata=metadata
            )
            
            self.logger.debug(f"Successfully processed: {file_path} in {processing_time:.2f}s")
            return result
            
        except Exception as e:
            processing_time = time.time() - start_time
            self.logger.error(f"Failed to process {file_path}: {str(e)}")
            
            return ProcessingResult(
                file_path=str(file_path),
                success=False,
                error=str(e),
                processing_time=processing_time
            )
    
    def _extract_text_from_file(self, file_path: Path) -> str:
        """Extract text from non-PDF files."""
        if file_path.suffix.lower() == '.txt':
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        elif file_path.suffix.lower() == '.docx':
            # Would need python-docx library
            raise NotImplementedError("DOCX support not implemented")
        elif file_path.suffix.lower() == '.epub':
            # Would need ebooklib library
            raise NotImplementedError("EPUB support not implemented")
        else:
            raise ValueError(f"Unsupported file type: {file_path.suffix}")
    
    def _chunk_text(self, text: str) -> List[str]:
        """Chunk text into processing units."""
        # Simple chunking by character count
        # In production, would use more sophisticated chunking
        chunks = []
        chunk_size = self.config.chunk_size
        
        for i in range(0, len(text), chunk_size):
            chunk = text[i:i + chunk_size]
            if chunk.strip():
                chunks.append(chunk.strip())
        
        return chunks
    
    def _generate_output_filename(self, input_file: Path) -> Path:
        """Generate output filename for processed file."""
        output_dir = Path(self.config.output_path) / 'data'
        base_name = input_file.stem
        extension = f'.{self.config.format}'
        
        return output_dir / f"{base_name}_processed{extension}"
    
    def _prepare_export_data(self, chunks: List[str], enhanced_chunks: List[str],
                           quality_scores: List[float], source_file: Path) -> List[Dict[str, Any]]:
        """Prepare data for export."""
        export_data = []
        
        for i, (original, enhanced, quality) in enumerate(zip(chunks, enhanced_chunks, quality_scores)):
            item = {
                'id': f"{source_file.stem}_{i:04d}",
                'source_file': str(source_file),
                'chunk_index': i,
                'original_text': original,
                'enhanced_text': enhanced,
                'quality_score': quality,
                'processing_timestamp': datetime.now().isoformat(),
                'tone': self.config.tone,
                'meets_threshold': quality >= self.config.quality_threshold
            }
            export_data.append(item)
        
        return export_data
    
    def _generate_file_metadata(self, file_path: Path, extraction_metadata: Dict,
                              normalization_result: Dict, chunk_count: int,
                              avg_quality: float) -> Dict[str, Any]:
        """Generate metadata for processed file."""
        return {
            'source_file': {
                'path': str(file_path),
                'name': file_path.name,
                'size': file_path.stat().st_size,
                'modified': datetime.fromtimestamp(file_path.stat().st_mtime).isoformat(),
                'hash': self._calculate_file_hash(file_path)
            },
            'processing': {
                'timestamp': datetime.now().isoformat(),
                'config': asdict(self.config),
                'extraction_method': extraction_metadata.get('method', 'unknown'),
                'language_detected': normalization_result.get('language', 'unknown'),
                'language_confidence': normalization_result.get('confidence', 0.0)
            },
            'results': {
                'chunk_count': chunk_count,
                'average_quality': avg_quality,
                'chunks_above_threshold': sum(1 for score in [avg_quality] if score >= self.config.quality_threshold)
            },
            'version': {
                'processor_version': '1.0.0',
                'format_version': '1.0'
            }
        }
    
    def _calculate_file_hash(self, file_path: Path) -> str:
        """Calculate SHA-256 hash of file."""
        import hashlib
        
        hash_sha256 = hashlib.sha256()
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_sha256.update(chunk)
        
        return hash_sha256.hexdigest()
    
    def _dry_run_report(self, files: List[Path]) -> Dict[str, Any]:
        """Generate dry run report without processing."""
        total_size = sum(f.stat().st_size for f in files)
        
        return {
            'dry_run': True,
            'files_to_process': len(files),
            'total_size_bytes': total_size,
            'total_size_mb': total_size / (1024 * 1024),
            'estimated_time_minutes': len(files) * 2,  # Rough estimate
            'output_directory': self.config.output_path,
            'configuration': asdict(self.config),
            'files': [str(f) for f in files[:10]]  # Show first 10 files
        }
    
    def _generate_final_report(self, results: List[ProcessingResult]) -> Dict[str, Any]:
        """Generate final processing report."""
        successful = [r for r in results if r.success]
        failed = [r for r in results if not r.success]
        
        total_time = time.time() - self.start_time
        total_chunks = sum(r.chunks_processed for r in successful)
        avg_quality = sum(r.quality_score for r in successful) / len(successful) if successful else 0
        
        return {
            'summary': {
                'total_files': len(results),
                'successful': len(successful),
                'failed': len(failed),
                'success_rate': len(successful) / len(results) if results else 0,
                'total_processing_time': total_time,
                'total_chunks_processed': total_chunks,
                'average_quality_score': avg_quality
            },
            'configuration': asdict(self.config),
            'results': [asdict(r) for r in results],
            'failed_files': [r.file_path for r in failed],
            'performance': {
                'files_per_minute': len(results) / (total_time / 60) if total_time > 0 else 0,
                'chunks_per_minute': total_chunks / (total_time / 60) if total_time > 0 else 0
            },
            'timestamp': datetime.now().isoformat()
        }
    
    def _save_processing_state(self, report: Dict[str, Any]):
        """Save processing state and report."""
        state_dir = Path(self.config.output_path) / 'state'
        
        # Save final report
        report_file = state_dir / 'final_report.json'
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        # Save configuration
        config_file = state_dir / 'config.json'
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(asdict(self.config), f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Processing state saved to: {state_dir}")
    
    def _save_intermediate_state(self, results: List[ProcessingResult]):
        """Save intermediate processing state."""
        state_dir = Path(self.config.output_path) / 'state'
        
        intermediate_file = state_dir / 'intermediate_state.json'
        state = {
            'processed_files': [r.file_path for r in results if r.success],
            'failed_files': [r.file_path for r in results if not r.success],
            'timestamp': datetime.now().isoformat(),
            'results_count': len(results)
        }
        
        with open(intermediate_file, 'w', encoding='utf-8') as f:
            json.dump(state, f, indent=2, ensure_ascii=False)
    
    def _load_resume_state(self):
        """Load previous processing state for resume."""
        state_dir = Path(self.config.output_path) / 'state'
        intermediate_file = state_dir / 'intermediate_state.json'
        
        if intermediate_file.exists():
            with open(intermediate_file, 'r', encoding='utf-8') as f:
                state = json.load(f)
            
            self.processed_files = set(state.get('processed_files', []))
            self.failed_files = set(state.get('failed_files', []))
            
            self.logger.info(f"Resumed from previous state: {len(self.processed_files)} files already processed")
    
    def _generate_interruption_report(self) -> Dict[str, Any]:
        """Generate report for interrupted processing."""
        return {
            'interrupted': True,
            'processed_files': len(self.processed_files),
            'failed_files': len(self.failed_files),
            'results': [asdict(r) for r in self.results],
            'timestamp': datetime.now().isoformat(),
            'message': 'Processing was interrupted. Use --resume to continue.'
        }
    
    def _signal_handler(self, signum, frame):
        """Handle interrupt signals gracefully."""
        self.logger.warning(f"Received signal {signum}, shutting down gracefully...")
        self.interrupted = True

def create_cli_parser() -> argparse.ArgumentParser:
    """Create command-line argument parser."""
    parser = argparse.ArgumentParser(
        description='Enhanced Universal AI Training Data Creator - CLI Batch Processor',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process single PDF file
  python enhanced_app.py --input book.pdf --output results/

  # Process directory of files
  python enhanced_app.py --input books/ --output results/ --format jsonl

  # Batch processing with custom settings
  python enhanced_app.py --input books/ --output results/ --tone zen_buddhism --quality-threshold 0.8 --workers 8

  # Resume interrupted processing
  python enhanced_app.py --input books/ --output results/ --resume

  # Dry run to see what would be processed
  python enhanced_app.py --input books/ --output results/ --dry-run
        """
    )
    
    # Required arguments
    parser.add_argument('--input', '-i', required=True,
                       help='Input file or directory path')
    parser.add_argument('--output', '-o', required=True,
                       help='Output directory path')
    
    # Processing options
    parser.add_argument('--format', '-f', default='jsonl',
                       choices=['jsonl', 'json', 'csv', 'xlsx', 'txt'],
                       help='Output format (default: jsonl)')
    parser.add_argument('--tone', '-t', default='universal_wisdom',
                       choices=['universal_wisdom', 'advaita_vedanta', 'zen_buddhism', 
                               'sufi_mysticism', 'christian_mysticism', 'mindfulness_meditation'],
                       help='Enhancement tone (default: universal_wisdom)')
    parser.add_argument('--quality-threshold', '-q', type=float, default=0.7,
                       help='Quality threshold for filtering (default: 0.7)')
    parser.add_argument('--chunk-size', '-c', type=int, default=1000,
                       help='Text chunk size in characters (default: 1000)')
    parser.add_argument('--workers', '-w', type=int, default=4,
                       help='Number of parallel workers (default: 4)')
    
    # Control options
    parser.add_argument('--resume', '-r', action='store_true',
                       help='Resume from previous interrupted processing')
    parser.add_argument('--overwrite', action='store_true',
                       help='Overwrite existing output files')
    parser.add_argument('--dry-run', '-n', action='store_true',
                       help='Show what would be processed without actually processing')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose logging')
    parser.add_argument('--no-metadata', action='store_true',
                       help='Skip metadata file generation')
    
    return parser

def main():
    """Main CLI entry point."""
    parser = create_cli_parser()
    args = parser.parse_args()
    
    # Create configuration from arguments
    config = BatchConfig(
        input_path=args.input,
        output_path=args.output,
        format=args.format,
        tone=args.tone,
        quality_threshold=args.quality_threshold,
        chunk_size=args.chunk_size,
        max_workers=args.workers,
        resume=args.resume,
        overwrite=args.overwrite,
        verbose=args.verbose,
        dry_run=args.dry_run,
        include_metadata=not args.no_metadata
    )
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(Path(args.output) / 'logs' / 'batch_processing.log')
        ]
    )
    
    try:
        # Create and run batch processor
        processor = CLIBatchProcessor(config)
        result = processor.process_batch()
        
        # Print summary
        if config.dry_run:
            print(f"\n🔍 DRY RUN SUMMARY:")
            print(f"Files to process: {result['files_to_process']}")
            print(f"Total size: {result['total_size_mb']:.1f} MB")
            print(f"Estimated time: {result['estimated_time_minutes']} minutes")
        else:
            print(f"\n✅ BATCH PROCESSING COMPLETE:")
            print(f"Files processed: {result['summary']['successful']}/{result['summary']['total_files']}")
            print(f"Success rate: {result['summary']['success_rate']:.1%}")
            print(f"Total time: {result['summary']['total_processing_time']:.1f} seconds")
            print(f"Average quality: {result['summary']['average_quality_score']:.2f}")
            print(f"Output directory: {config.output_path}")
        
        return 0
        
    except KeyboardInterrupt:
        print("\n⚠️  Processing interrupted by user")
        return 1
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        return 1

if __name__ == '__main__':
    sys.exit(main())

