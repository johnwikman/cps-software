#!/usr/bin/env python3

import cps_server

if __name__ == "__main__":
    #cps_server.connection.run_spider()
    #cps_server.run_policy.run_policy("/mnt/model")
    cps_server.interactive_policy.run_interactive_loop("/mnt/model_walk", "/mnt/model_creep", "/mnt/model_idle")
