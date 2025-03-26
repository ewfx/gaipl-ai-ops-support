CREATE TABLE incidents (
    record_id SERIAL PRIMARY KEY,
    record_type VARCHAR(50) NOT NULL CHECK (record_type IN ('incident', 'knowledge_article', 'history')),
    number VARCHAR(50) UNIQUE,  -- Only applicable for incidents/knowlege base
    title VARCHAR(255),  -- For knowledge articles
    short_description TEXT,  -- For incidents
    description TEXT,
    priority INT CHECK (priority BETWEEN 1 AND 5),  -- Only for incidents
    state VARCHAR(50),  -- Only for incidents
    assigned_to VARCHAR(100),  -- Only for incidents
    content TEXT,  -- Only for knowledge articles
    action VARCHAR(255),  -- Only for history
    old_value TEXT,  -- Only for history
    new_value TEXT,  -- Only for history
    updated_by VARCHAR(100),  -- Only for history
    created_by VARCHAR(100),  -- For knowledge articles
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
