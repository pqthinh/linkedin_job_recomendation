import React from 'react';
import JobItem from './JobItem';

const JobList = ({ jobs }) => (
  <div>
    {jobs.map((job, index) => (
      <JobItem key={index} job={job} />
    ))}
  </div>
);

export default JobList;