from __future__ import annotations
import hashlib, json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

@dataclass(frozen=True)
class FailureExample:
    example_id: str
    prompt: str
    teacher_output: str
    student_output: str
    task_type: str
    failure_type: str
    metadata: dict[str, Any]

def stable_id(prompt: str, teacher: str, student: str) -> str:
    return hashlib.sha256((prompt+'\n'+teacher+'\n'+student).encode()).hexdigest()[:20]

class FailureBuffer:
    def __init__(self, path: Path): self.path=path; self.path.parent.mkdir(parents=True,exist_ok=True)
    def read(self) -> list[FailureExample]:
        if not self.path.exists(): return []
        return [FailureExample(**json.loads(x)) for x in self.path.read_text(encoding='utf-8').splitlines() if x.strip()]
    def append(self, item: FailureExample) -> bool:
        if item.example_id in {x.example_id for x in self.read()}: return False
        with self.path.open('a',encoding='utf-8') as h: h.write(json.dumps(item.__dict__,ensure_ascii=False)+'\n')
        return True
    def extend_from_comparison(self, rows: Iterable[dict[str,Any]]) -> int:
        n=0
        for row in rows:
            if row.get('teacher_success') is True and row.get('student_success') is False:
                prompt=row.get('prompt',''); teacher=row.get('teacher_output',''); student=row.get('student_output','')
                n += self.append(FailureExample(stable_id(prompt,teacher,student),prompt,teacher,student,row.get('task_type','unknown'),row.get('failure_type','quantization_regression'),row.get('metadata',{})))
        return n
