from pulse_lib.uploader.uploader_funcs import merge_markers

def test_merge_markers():
    on_off = [
        (180, +1), (300, -1),
        (330, +1), (400, -1),
        (405, +1), (500, -1),
        (530, +1), (600, -1),
        (650, +1), (700, -1),
        (630, +1), (650, -1),
        (0, +1), (100, -1),
        (120, +1), (200, -1),
        ]

    print(merge_markers('test', on_off))


if __name__ == '__main__':
    test_merge_markers()