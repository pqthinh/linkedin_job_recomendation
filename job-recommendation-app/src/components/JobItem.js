import React from 'react';
import { Card, Button } from 'react-bootstrap';

const JobItem = ({ job }) => (
  <Card className="mb-3">
    <Card.Body>
      <Card.Title>{job.title}</Card.Title>
      <Card.Subtitle className="mb-2 text-muted">{job.company}</Card.Subtitle>
      <Card.Text>
        {job.location}
      </Card.Text>
      <Card.Text>
        {job.description}
      </Card.Text>
      <Button variant="primary">Apply Now</Button>
    </Card.Body>
  </Card>
);

export default JobItem;