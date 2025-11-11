import axios from "axios";

const client = axios.create({
  baseURL: "/api",
  timeout: 60000
});

export default client;
