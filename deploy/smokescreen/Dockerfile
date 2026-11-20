# smokescreen (github.com/stripe/smokescreen), the egress proxy the AI worker
# reaches the internet through (docker-compose.production.yml). Built from a
# pinned release: the clone must be exactly the commit the tag named when it
# was reviewed, or the build stops. Its dependencies are vendored in the repo.
FROM golang:1.27-alpine AS build
RUN apk add --no-cache git
ARG SMOKESCREEN_VERSION=v0.1.0
ARG SMOKESCREEN_COMMIT=609eb8931420453daf5893509be0b25b21bd9edb
RUN git clone --quiet --depth 1 --branch "$SMOKESCREEN_VERSION" https://github.com/stripe/smokescreen.git /src \
 && test "$(git -C /src rev-parse HEAD)" = "$SMOKESCREEN_COMMIT"
WORKDIR /src
RUN CGO_ENABLED=0 go build -mod=vendor -trimpath -ldflags="-s -w" -o /out/smokescreen .

FROM alpine:3.22
RUN apk upgrade --no-cache && adduser -D -H -u 10002 smokescreen
COPY --from=build /out/smokescreen /usr/local/bin/smokescreen
COPY config.yaml acl.yaml /etc/smokescreen/
USER smokescreen
EXPOSE 4750
ENTRYPOINT ["/usr/local/bin/smokescreen", "--config-file", "/etc/smokescreen/config.yaml"]
