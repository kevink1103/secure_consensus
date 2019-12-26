def count(tenders, target, memory):
    '''a function to count the number of tender combitnations that meets target'''
    if sum(memory) == target:
        print(memory)
        return 1
    elif sum(memory) > target:
        return 0
    else:
        counts = 0
        for tender in tenders[::-1]:
            if len(memory) > 0 and tender < memory[-1]:
                continue
            new_mem = memory.copy()
            new_mem.append(tender)
            counts += count(tenders, target, new_mem)
        return counts

def main():
    '''main function of this program'''
    tenders = [100, 50, 20 , 10, 5, 2, 1]
    target = 10
    
    # run recursive task
    poss = count(tenders, target, [])
    print(poss)

if __name__ == "__main__":
    main()
