/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  experimental: {
    // /evaluate runs the full ragas suite; the default 30s rewrite proxy timeout is too short.
    proxyTimeout: 180_000,
  },
  async rewrites() {
    return [
      {
        source: '/api/:path*',
        destination: 'http://127.0.0.1:8000/:path*',
      },
    ];
  },
};

module.exports = nextConfig;
