from dataclasses import dataclass, field

@dataclass
class Ledger:
    entries:list=field(default_factory=list)
    def add(self,kind,amount,symbol='',reference=''): self.entries.append({'kind':kind,'amount':float(amount),'symbol':symbol,'reference':reference})
    def snapshot(self): return list(self.entries)
    @classmethod
    def restore(cls,entries): return cls(list(entries or []))
