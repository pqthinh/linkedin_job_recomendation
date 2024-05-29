import React, { useState } from 'react';
import { Form, Button, Col, Row } from 'react-bootstrap';

const SearchForm = ({ onSearch }) => {
  const [skills, setSkills] = useState('');
  const [location, setLocation] = useState('');
  const [salary, setSalary] = useState('');
  const [jobTitle, setJobTitle] = useState('')

  const handleSubmit = (e) => {
    e.preventDefault();
    onSearch({ skills, location, salary, jobTitle });
  };

  return (
    <Form onSubmit={handleSubmit}>
      <Row>
        <Form.Group as={Col} controlId="formGridSkills">
          <Form.Label>Skills</Form.Label>
          <Form.Control
            type="text"
            placeholder="Enter your skills"
            value={skills}
            onChange={(e) => setSkills(e.target.value)}
          />
        </Form.Group>

        <Form.Group as={Col} controlId="formGridJobTitle">
          <Form.Label>JobTitle</Form.Label>
          <Form.Control
            type="text"
            placeholder="Enter preferred location"
            value={jobTitle}
            onChange={(e) => setJobTitle(e.target.value)}
          />
        </Form.Group>

        <Form.Group as={Col} controlId="formGridSalary">
          <Form.Label>Salary</Form.Label>
          <Form.Control
            type="number"
            placeholder="Enter desired salary"
            value={salary}
            onChange={(e) => setSalary(e.target.value)}
          />
        </Form.Group>

        <Form.Group as={Col} controlId="formGridLocation">
          <Form.Label>Location</Form.Label>
          <Form.Control
            type="text"
            placeholder="Enter location"
            value={location}
            aria-describedby="basic-addon2"
            onChange={(e) => setLocation(e.target.value)}
          />
        </Form.Group>
        
      </Row>
      <Row className='flex-row-reverse m-1'>
        <Button variant="primary" type="submit" size='small' className='m-2 col-2'>
          Search
        </Button>
      </Row>
      
    </Form>
  );
};

export default SearchForm;