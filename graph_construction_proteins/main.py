"""
Main pipeline to construct protein graphs from a CSV of UniProt entries.
Loads a protein CSV and generates PyTorch graphs in the output folder.
"""

import os
import sys
import pandas as pd
import torch
import logging
from pathlib import Path
from tqdm import tqdm
from datetime import datetime
from typing import Optional

# Add parent directory to path
sys.path.insert(0, os.path.dirname(__file__))
from functions import construct_protein_graph_from_uniprot, print_graph_info

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ProteinGraphPipeline:
    """Pipeline for constructing and saving protein graphs."""
    
    def __init__(self, input_csv: str, output_dir: str = "graph proteins", distance_cutoff: float = 10.0):
        """
        Initialize the pipeline.
        
        Args:
            input_csv: Path to CSV file containing protein entries and UniProt IDs
            output_dir: Directory to save protein graphs
            distance_cutoff: Maximum distance in Angstroms for edges (default: 10.0 A)
        """
        self.input_csv = input_csv
        self.output_dir = Path(output_dir)
        self.distance_cutoff = distance_cutoff
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.stats = {
            'total_proteins': 0,
            'graphs_created': 0,
            'graphs_failed': 0
        }
        
        logger.info(f"Pipeline initialized. Output directory: {self.output_dir}")
        logger.info(f"Distance cutoff for edges: {distance_cutoff}A")
    
    def load_dataset(self) -> pd.DataFrame:
        """Load the protein dataset CSV."""
        try:
            df = pd.read_csv(self.input_csv)
            df = df.drop_duplicates(subset="protein_accession")
            
            logger.info(f"Loaded {len(df)} proteins from {self.input_csv}")
            return df
        except FileNotFoundError:
            logger.error(f"Input CSV not found: {self.input_csv}")
            return None

    def get_protein_id(self, row: pd.Series) -> Optional[str]:
        """Extract the UniProt accession ID from a dataset row."""
        for col in ['UniProt_ID', 'UniProt ID', 'uniprot_id', 'uniprot id', 'Entry', 'entry' , 'protein_accession']:
            if col in row and pd.notna(row[col]):
                return str(row[col]).strip()
        if len(row) > 0:
            return str(row.iloc[0]).strip()
        return None

    def get_output_path(self, entry: str) -> Path:
        """
        Get the output path for a protein graph.
        
        Args:
            entry: UniProt entry or UniProt accession ID
        Returns:
            Path to save the graph
        """
        return self.output_dir / f"{entry}.pt"
    
    def process_protein(self, entry: str) -> bool:
        """
        Process a single protein and save its graph.
        
        Args:
            entry: UniProt accession ID
        Returns:
            True if successful, False otherwise
        """
        output_path = self.get_output_path(entry)
        
        # Skip if already exists
        if output_path.exists():
            logger.info(f"Graph already exists: {entry}")
            return True
        
        # Construct graph with distance cutoff
        graph = construct_protein_graph_from_uniprot(entry, distance_cutoff=self.distance_cutoff)
        
        if graph is None:
            logger.warning(f"Failed to construct graph for {entry}")
            self.stats['graphs_failed'] += 1
            return False
        
        # Save graph
        try:
            torch.save(graph, output_path)
            logger.info(f"Saved graph for {entry} to {output_path}")
            self.stats['graphs_created'] += 1
            return True
        except Exception as e:
            logger.error(f"Error saving graph for {entry}: {e}")
            self.stats['graphs_failed'] += 1
            return False
    
    def run(self, verbose: bool = False, limit: Optional[int] = None) -> None:
        """
        Run the complete pipeline.
        
        Args:
            verbose: Print detailed graph information
            limit: Maximum number of proteins to process (None for all)
        """
        logger.info("Starting protein graph construction pipeline...")
        start_time = datetime.now()
        
        # Load dataset
        df = self.load_dataset()
        if df is None:
            return
        
        self.stats['total_proteins'] = len(df)
        
        # Process limit
        if limit is not None and limit < len(df):
            df = df.head(limit)
            logger.info(f"Processing limited to {limit} proteins")
        
        # Process each protein
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing proteins"):
            entry = self.get_protein_id(row)
            if entry is None or entry == '':
                logger.warning(f"Skipping row {idx} because no UniProt ID was found")
                self.stats['graphs_failed'] += 1
                continue

            success = self.process_protein(entry)
            
            if success and verbose:
                # Load and print graph info
                output_path = self.get_output_path(entry)
                graph = torch.load(output_path)
                print_graph_info(graph, entry)
        
        # Print final statistics
        self.print_statistics(start_time)
    
    def print_statistics(self, start_time) -> None:
        """Print pipeline execution statistics."""
        elapsed = datetime.now() - start_time
        
        print("\n" + "="*70)
        print("PROTEIN GRAPH CONSTRUCTION PIPELINE - STATISTICS")
        print("="*70)
        print(f"Total time: {elapsed}")
        print(f"\nProcessed proteins: {self.stats['total_proteins']}")
        print(f"Successfully created graphs: {self.stats['graphs_created']}")
        print(f"Failed graphs: {self.stats['graphs_failed']}")
        print(f"Success rate: {100 * self.stats['graphs_created'] / max(1, self.stats['total_proteins']):.1f}%")
        print(f"\nOutput directory: {self.output_dir}")
        print("="*70 + "\n")


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Construct protein graphs from a CSV with UniProt IDs"
    )
    parser.add_argument(
        '--input',
        type=str,
        default='final/combined_active_inactive_rows.csv',
        help='Path to merged UniProt/InterPro CSV file'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='graph_construction_proteins/graphproteins',
        help='Output directory for protein graphs'
    )
    parser.add_argument(
        '--distance-cutoff',
        type=float,
        default=10.0,
        help='Maximum distance in Angstroms to consider as edge (default: 10.0)'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Print detailed information for each graph'
    )
    parser.add_argument(
        '--limit',
        type=int,
        default=None,
        help='Limit number of proteins to process (for testing)'
    )
    
    args = parser.parse_args()
    
    # Create and run pipeline
    pipeline = ProteinGraphPipeline(
        input_csv=args.input,
        output_dir=args.output,
        distance_cutoff=args.distance_cutoff
    )
    
    pipeline.run(verbose=args.verbose, limit=args.limit)


if __name__ == "__main__":
    main()
