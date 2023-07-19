import gpac

def main():
    # reversible unimolecular A <--> B

    a,b = gpac.species('A B')
    rxns = [a | b]
    for rxn in rxns:
        print(rxn)

    inits = {a: 1}
    print(f'inits: {inits}')

    odes = gpac.crn_to_odes(rxns)
    print(odes)


if __name__ == '__main__':
    main()