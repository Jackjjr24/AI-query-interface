CREATE DATABASE woodpecker;
USE woodpecker;

CREATE TABLE clients (
    id INT AUTO_INCREMENT PRIMARY KEY,
    name VARCHAR(100),
    country VARCHAR(100),
    state VARCHAR(100),
    email VARCHAR(100)
);

INSERT INTO clients (name, country, state, email) VALUES
('John Doe', 'New York', 'NY', 'john.doe@example.com'),
('Jane Smith', 'Los Angeles', 'CA', 'jane.smith@example.com'),
('Alice Johnson', 'Chicago', 'IL', 'alice.johnson@example.com');
