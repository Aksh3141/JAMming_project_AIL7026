import sys
import random

def generate_dataset(universal_range, num_transactions, output_file):
    try:
        universal_range = 300 # fixing universal range - number of items in itemset
        num_items = int(universal_range)
        universal_items = [str(i) for i in range(1, num_items + 1)]
    except ValueError:
        try:
            start, end = map(int, universal_range.split('-'))
            universal_items = [str(i) for i in range(start, end + 1)]
        except ValueError:
            universal_items = universal_range.split()

    num_transactions = int(num_transactions)
    core_percent = 0.07
    core_size = int(core_percent*num_items)
    core_items = universal_items[:core_size]
    
    # Add a "semi-frequent" tier - items that are frequent at 5% but not at 10%
    semi_freq_percent = 0.03
    semi_freq_size = int(semi_freq_percent * num_items)
    semi_freq_items = universal_items[core_size:core_size + semi_freq_size]
    
    other_items = universal_items[core_size + semi_freq_size:]

    with open(output_file, 'w') as f:
        for _ in range(num_transactions):
            transaction = []
            
            # Core items: very frequent (appear in ~95% of transactions)
            for item in core_items:
                if random.random() < 0.95:
                    transaction.append(item)
            
            # Semi-frequent items: appear in ~7-8% of transactions
            # These create extra work at 5% support but get filtered at 10%+
            for item in semi_freq_items:
                if random.random() < 0.06:
                    transaction.append(item)
            
            # Noise items
            if other_items:
                noise_count = random.randint(1, 4)
                transaction.extend(random.sample(other_items, min(noise_count, len(other_items))))
            
            random.shuffle(transaction)
            f.write(" ".join(transaction) + "\n")

if __name__ == "__main__":
    # The bash script passes: <universal_itemset> <num_transactions> <output_file>
    generate_dataset(sys.argv[1], sys.argv[2], sys.argv[3])
