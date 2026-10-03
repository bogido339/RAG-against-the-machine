def floate(value):
    res = 0
    res2 = 0
    flg = True

    for c in value:
        if c.isnumeric():
            if flg:
                res = res * 10 + int(c)
            else:
                res2 = res2 * 10 + int(c)
        elif c == '.' and flg:
            flg = False
        else:
            raise ValueError(f"{value} is not a valid number")

    return res + res2 / 10

print(floate("12..2"))