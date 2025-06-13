#!/bin/bash

docker run --rm --name redis -p 6379:6379 redis --requirepass torchpass
docker exec -it redis redis-cli --pass torchpass --stat
