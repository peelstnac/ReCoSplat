"""Chunk schedule for an autoregressive view stream."""

from dataclasses import dataclass


@dataclass(frozen=True)
class ChunkScheduler:
    chunk_size_f: int
    chunk_size_s: int

    def __post_init__(self) -> None:
        if self.chunk_size_f < 1 or self.chunk_size_s < 1:
            raise ValueError(
                f"chunk sizes must be >= 1, got ({self.chunk_size_f}, {self.chunk_size_s})"
            )

    def intervals(self, num_views: int) -> list[tuple[int, int]]:
        """Half-open view intervals [start, end) covering range(num_views), in order."""
        out: list[tuple[int, int]] = []
        start = 0
        while start < num_views:
            size = self.chunk_size_f if start == 0 else self.chunk_size_s
            end = min(start + size, num_views)
            out.append((start, end))
            start = end
        return out

    def num_chunks(self, num_views: int) -> int:
        return len(self.intervals(num_views))

    def chunk_assignments(
        self, num_views: int
    ) -> tuple[list[int], dict[int, int], dict[int, int]]:
        """Per-view chunk ids plus the first/last view index of each chunk.

        Returns:
            chunk_ids_per_view: len num_views, chunk index for each view
            first_view_by_chunk: chunk_id -> first view idx
            last_view_by_chunk: chunk_id -> last view idx (inclusive)
        """
        chunk_ids_per_view: list[int] = []
        first_view_by_chunk: dict[int, int] = {}
        last_view_by_chunk: dict[int, int] = {}
        for chunk_id, (start, end) in enumerate(self.intervals(num_views)):
            first_view_by_chunk[chunk_id] = start
            last_view_by_chunk[chunk_id] = end - 1
            chunk_ids_per_view.extend([chunk_id] * (end - start))
        assert len(chunk_ids_per_view) == num_views, "chunk assignment does not cover all views"
        return chunk_ids_per_view, first_view_by_chunk, last_view_by_chunk
