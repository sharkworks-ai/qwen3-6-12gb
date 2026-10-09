# Extra root CAs

Place PEM-encoded `*.crt` root certificates here to trust them inside the worker
image, for example when the build host sits behind a TLS-inspecting proxy.
`*.crt` files in this directory are gitignored and must not be committed.
