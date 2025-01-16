from agent_attribute import ChangingAttribute, ConstantValue, Iterator

def main():
    print("Testing ChangingAttribute:")
    try:
        ca = ChangingAttribute(value=10, min_value=0)
        print(f"Initial value: {ca.value}")
        ca.change_value(5)
        print(f"After adding 5: {ca.value}")
        ca.change_value(-8)
        print(f"After subtracting 8: {ca.value}")
        # This should raise an exception
        ca.change_value(-10)
    except ValueError as e:
        print(f"Caught expected exception: {e}")

    print("\nTesting ConstantValue:")
    try:
        cv = ConstantValue(value=100)
        print(f"Constant value: {cv.value}")
        # This should raise an exception
        cv.change_value(10)
    except AttributeError as e:
        print(f"Caught expected exception: {e}")

    print("\nTesting Iterator:")
    try:
        it = Iterator(value=0, min_value=0, max_value=20, delta=-2)
        for _ in range(6):
            it.change_value()
            print(f"Iterator value: {it.value}")
        # This should raise an exception
        it.change_value()
    except ValueError as e:
        print(f"Caught expected exception: {e}")

if __name__ == "__main__":
    main()
