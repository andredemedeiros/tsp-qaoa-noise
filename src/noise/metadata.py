# def get_noise_metadata() -> dict:
#     """
#     Metadata for visualization and reporting.
#     """
#     return {
#         "ideal": {
#             "label": "Ideal (No Noise)",
#             "color": "seagreen",
#             "desc": "error-free",
#         },
#         "depolarizing": {
#             "label": "Depolarizing",
#             "color": "tomato",
#             "desc": "p1q=0.005, p2q=0.02",
#         },
#         "bit_flip": {
#             "label": "Bit-Flip",
#             "color": "mediumorchid",
#             "desc": "p=0.01",
#         },
#         "phase_flip": {
#             "label": "Phase-Flip",
#             "color": "darkorange",
#             "desc": "p=0.01",
#         },
#         "thermal": {
#             "label": "Thermal Relaxation",
#             "color": "steelblue",
#             "desc": "T1=50µs, T2=30µs",
#         },
#     }