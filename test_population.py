from member import MemberState, Population

population_size = 10
members = [MemberState(i, None, False) for i in range(107)]
population = Population(size=population_size)

for member in members:
    if population.full():
        population.new_generation()
    population.append(member)

print("population size", population.size)
print("Number of members", len(population))
print("Number of generations", len(population.generations))
print("Number of members across generations", len(population.members))

print("Population:")
print(",".join(str(m.id) for m in population))

print("Previous generation:")
print(",".join(str(m.id) for m in population.generations[-2]))

print("Adding 3 new members:")
population.append(MemberState(1123, None, False))
population.append(MemberState(1456, None, False))
population.append(MemberState(1789, None, False))
print(",".join(str(m.id) for m in population))

print("Removing 2 of the new members:")
population.remove(MemberState(1456, None, False))
population.remove(MemberState(1789, None, False))
print(",".join(str(m.id) for m in population))