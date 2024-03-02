from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict

import gpac


class InstructionType(Enum):
    inc = 'inc'
    dec = 'dec'
    goto = 'goto'
    halt = 'halt'


def inc(register: str) -> Instruction:
    return Instruction(instruction_type=InstructionType.inc, register=register, branch=0)


def dec(register: str, branch: int) -> Instruction:
    return Instruction(instruction_type=InstructionType.dec, register=register, branch=branch)


def goto(branch: int) -> Instruction:
    return Instruction(instruction_type=InstructionType.goto, register='', branch=branch)


def halt() -> Instruction:
    return Instruction(instruction_type=InstructionType.halt, register='', branch=0)


@dataclass
class Instruction:
    instruction_type: InstructionType
    register: str
    branch: int

    def __str__(self) -> str:
        if self.instruction_type == InstructionType.inc:
            return f'inc {self.register}'
        elif self.instruction_type == InstructionType.dec:
            return f'dec {self.register},{self.branch}'
        elif self.instruction_type == InstructionType.goto:
            return f'goto {self.branch}'
        elif self.instruction_type == InstructionType.halt:
            return 'halt'


@dataclass
class RegisterMachine:
    instructions: Dict[int, Instruction]
    registers: List[str] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        if list(self.instructions.keys()) != list(range(1, len(self.instructions) + 1)):
            raise AssertionError('keys of instructions must be integers 1,2,...,m')

        for instr in self.instructions.values():
            if instr.instruction_type == InstructionType.dec:
                if instr.branch not in self.instructions.keys():
                    raise AssertionError(f'line {instr.branch} is not a valid line in the program')

        for instr in self.instructions.values():
            if instr.register and instr.register not in self.registers:
                self.registers.append(instr.register)

        self.registers.sort()

        last_instruction = list(self.instructions.values())[-1]
        if last_instruction.instruction_type != InstructionType.halt:
            raise AssertionError('last instruction must be halt')

    def __str__(self) -> str:
        return '\n'.join([str(instr) for instr in self.instructions])

    def __repr__(self) -> str:
        return str(self)

    def execute(self, initial_registers: Dict[str, int]) -> int:
        registers = {register: initial_registers.get(register, 0) for register in self.registers}
        line = 1
        last_line = max(self.instructions.keys())
        while line < last_line + 1:
            instr = self.instructions[line]
            if instr.instruction_type == InstructionType.inc:
                registers[instr.register] += 1
                line += 1
            elif instr.instruction_type == InstructionType.dec:
                if registers[instr.register] == 0:
                    line = instr.branch
                else:
                    registers[instr.register] -= 1
                    line += 1
            elif instr.instruction_type == InstructionType.goto:
                line = instr.branch
            elif instr.instruction_type == InstructionType.halt:
                break
            else:
                raise AssertionError(f'unknown instruction type {instr.instruction_type}')
        register_values = list(registers.values())
        return register_values[-1]

    def to_icrn(self, inhibitor_constant: float) -> List[gpac.Reaction]:
        predecessors = {line: [] for line in self.instructions.keys()}
        max_line = max(self.instructions.keys())
        for line, instr in self.instructions.items():
            if instr.instruction_type in [InstructionType.inc, InstructionType.dec] and line < max_line:
                predecessors[line + 1].append(line)
                if instr.instruction_type == InstructionType.dec:
                    predecessors[instr.branch].append(line)
            elif instr.instruction_type == InstructionType.goto:
                predecessors[instr.branch].append(line)
            elif instr.instruction_type == InstructionType.halt:
                pass
            else:
                raise AssertionError(f'unknown instruction type {instr.instruction_type}')

        a_species = {line: gpac.Specie(f'A{line}') for line in self.instructions.keys()}
        b_species = {line: gpac.Specie(f'B{line}') for line in self.instructions.keys()}
        c_species = {line: gpac.Specie(f'C{line}') for line in self.instructions.keys()}

        register_species = {register: gpac.Specie(register) for register in self.registers}
        for register in self.registers:
            if register in list(a_species.values()) + list(b_species.values()) + list(c_species.values()):
                raise ValueError(f'cannot use name "{register}" for register sicne it conflicts with '
                                 f'the name of the clock species')

        rxns = []
        for line, instr in list(self.instructions.items())[:-1]:
            if instr.instruction_type == InstructionType.halt:
                continue

            a = a_species[line]
            b = b_species[line]
            bp1 = b_species[line + 1]
            c = c_species[line]

            if instr.instruction_type == InstructionType.inc:
                r = register_species[instr.register]
                rxns.append((a >> bp1 + r).with_inhibitor(c, inhibitor_constant))

            elif instr.instruction_type == InstructionType.dec:
                r = register_species[instr.register]
                rxns.append((a + r >> bp1).with_inhibitor(c, inhibitor_constant))
                b_branch = b_species[instr.branch]
                rxns.append((a >> b_branch).with_inhibitor(c, inhibitor_constant).with_inhibitor(r, inhibitor_constant))

            elif instr.instruction_type == InstructionType.goto:
                b_branch = b_species[instr.branch]
                rxns.append((a >> b_branch).with_inhibitor(c, inhibitor_constant))

            b_rxn = b >> c
            for pre in predecessors[line]:
                a_pred = a_species[pre]
                b_rxn.with_inhibitor(a_pred, inhibitor_constant)
            rxns.append(b_rxn)
            rxns.append((c >> a).with_inhibitor(b, inhibitor_constant))

        return rxns


def main():
    instructions = {
        1: dec('x', 4),
        2: inc('y'),
        3: goto(1),
        4: halt(),
    }
    for line, instr in instructions.items():
        print(f'{line}. {instr}')
    rm = RegisterMachine(instructions)
    # a = 10
    # output = rm.execute({'a': a})
    # print(f'output = {output} on input a = {a}')
    rxns = rm.to_icrn(1000)
    print('rxns to simulate register machine:')
    for rxn in rxns:
        print(rxn)


if __name__ == '__main__':
    main()
