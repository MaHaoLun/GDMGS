"""One sorted union, one source decode, and target membership over shared rows."""
from dataclasses import dataclass
import torch


def union_plan(requests):
    if not requests:
        raise ValueError('empty group')
    device = requests[0].device
    for ids in requests:
        if ids.dtype != torch.long or ids.ndim != 1 or ids.device != device:
            raise ValueError('IDs must be int64 vectors on the same device')
        if bool((ids < 0).any()) or (len(ids) > 1 and not bool((ids[1:] > ids[:-1]).all())):
            raise ValueError('IDs must be nonnegative, sorted and unique')
    union, inverse = torch.unique(torch.cat(requests), sorted=True, return_inverse=True)
    membership = torch.zeros((len(union), len(requests)), dtype=torch.bool, device=device)
    cursor = 0
    for target, ids in enumerate(requests):
        membership[inverse[cursor:cursor+len(ids)], target] = True
        cursor += len(ids)
    return union, membership


@dataclass(frozen=True)
class SharedRows:
    union_ids: torch.Tensor
    membership: torch.Tensor
    batch: object

    def row_mask(self, target):
        if not 0 <= target < self.membership.shape[1]:
            raise IndexError(target)
        owners = self.batch.bundle_metadata.row_owner_ids
        if not len(self.union_ids):
            if len(owners):
                raise ValueError('nonempty rows for empty union')
            return torch.empty(0, dtype=torch.bool, device=self.union_ids.device)
        positions = torch.searchsorted(self.union_ids, owners)
        if bool((positions >= len(self.union_ids)).any()) or not torch.equal(self.union_ids[positions], owners):
            raise ValueError('row owners do not belong to this group')
        return self.membership[positions, target]


class GroupMaterializer:
    """Decoder protocol: prepare(source, ids), row_count(state), finish(state, ids).

    prepare evaluates opacity exactly once. finish consumes that same state.
    The scheduler reserves row_count before finish and releases after rendering.
    """
    def __init__(self, decoder):
        self.decoder = decoder

    def prepare(self, source, requests):
        union, membership = union_plan(requests)
        state = self.decoder.prepare(source, union)
        rows = self.decoder.row_count(state)
        if isinstance(rows, bool) or not isinstance(rows, int) or rows < 0:
            raise ValueError('decoder must report a nonnegative exact integer row count')
        return (union, membership, state), rows

    def finish(self, prepared):
        union, membership, state = prepared
        batch = self.decoder.finish(state, union)
        if len(batch.xyz) != self.decoder.row_count(state):
            raise ValueError('decoder output differs from reserved row count')
        if not torch.equal(batch.anchor_indices, union):
            raise ValueError('decoder changed the original anchor IDs')
        batch.validate_contract()
        return SharedRows(union, membership, batch)
